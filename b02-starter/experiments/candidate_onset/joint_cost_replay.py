"""Run a causal two-row reassignment before the ordinary track update."""
import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from b02.replay import _scans, open_scenes, replay_frames
from experiments.candidate_onset.dual_hypothesis import alternative
from experiments.candidate_onset.fallback import BackupTracker
from experiments.candidate_onset.mixed import install
from experiments.candidate_onset.propagation import MIN_AFFINITY, anchor, sha
from experiments.candidate_onset.update_stream import update_class
from experiments.id_recovery.affinity import pack
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.v2_association.run_stage1 import Diagnostic, save


@dataclass(frozen=True)
class Proposal:
    gain: float
    winner: int
    point: int
    rival: int
    other: int | None
    swap: bool
    other_published: bool


def apply_proposals(matches, proposals, cutoff):
    """Alter only disjoint pairs. Original point reservations stay in force."""
    touched = set()
    accepted = []
    for p in sorted(proposals, key=lambda item: (-item.gain, item.winner, item.rival)):
        if p.gain <= cutoff or {p.winner, p.rival} & touched:
            continue
        assert matches[p.winner] == p.point and matches.get(p.rival) == p.other
        matches[p.rival] = p.point
        if p.swap:
            matches[p.winner] = p.other
        else:
            del matches[p.winner]
        touched.update((p.winner, p.rival))
        accepted.append(p)
    assert len(set(matches.values())) == len(matches)
    return accepted


class JointCostTracker(BackupTracker):
    def __call__(self, points, scores):
        self.joint_proposals, self.joint_applied, self.joint_eligible = [], [], []
        self.joint_mature = {b for b in self.state.get('tracks', {}) if self.emissions.get(b, 0) >= 4
                             and len(self.history.get(b, ())) >= 4}
        return super().__call__(points, scores)

    def assign(self, track_ids, scan, positions, grid, occupied, points, scores):
        stage = self.assignment_stage
        matched = super().assign(track_ids, scan, positions, grid, occupied, points, scores)
        if stage:
            return matched
        first_rows = {}
        for event in self.events:
            first_rows.setdefault(event['batch'], event)
        events = {batch: e for batch, e in first_rows.items()
                  if 'effective_costs' in e and batch in self.joint_mature}
        eligible = [(b, i, events[b], events[b]['indices'].index(i)) for b, i in matched.items()
                    if b in events and i in events[b]['indices'] and scores[i] >= self.gate.policy.enter]
        if not eligible:
            return matched
        logits = self.affinity.score(pack(np.stack([e['track_features'] for _, _, e, _ in eligible]),
                                          np.stack([e['edge_features'][j] for _, _, e, j in eligible])))
        for (winner, index, _, _), logit in zip(eligible, logits, strict=True):
            if logit < MIN_AFFINITY:
                continue
            self.joint_eligible.append((winner, index))
            rival_choice = alternative(events, matched, self.joint_mature, winner, index)
            if rival_choice is None:
                continue
            gap, rival, old, swap = rival_choice
            other_published = old is not None and self.would_publish(rival, scores[old], scan)
            self.joint_proposals.append(Proposal(-float(gap), winner, index, rival, old, swap, other_published))
        if self.joint_cutoff is not None:
            self.joint_applied = apply_proposals(matched, self.joint_proposals, self.joint_cutoff)
        return matched

    def would_publish(self, batch, score, scan):
        policy = self.gate.policy
        last = self.gate.last_strong.get(batch)
        return score >= policy.enter or (policy.grace > 0 and last is not None
                                         and scan - 1 - last <= policy.grace and score >= policy.maintain)


def audit(p, points, labels, history):
    target = labels[points[p.point]['PntID']]
    winner = anchor(history.get(p.winner, ()))
    rival = anchor(history.get(p.rival, ()))
    primary = update_class(history.get(p.winner, ()), target)
    other = labels[points[p.other]['PntID']] if p.other is not None else None
    secondary_correct = p.other_published and rival is not None and other == rival
    # A swap that would put the other point on its own true track is a known
    # correctable error. Unknown secondary labels stay out of this count.
    helpful = primary == 2 and rival == target and (other is None or (other >= 0 and (not p.swap or other == winner)))
    return int(primary == 1) + int(secondary_correct), primary, helpful


def run(frames, scores, offsets, tracker, out, scene, shadow):
    labels_by_id = defaultdict(lambda: deque(maxlen=8))
    counts = Counter()
    controls, decisions = [], []
    diagnostic, births = Diagnostic(), BirthAudit()
    i = 0

    def handler(points, _):
        nonlocal i
        circle, _, labels = frames[i]
        scan_scores = scores[offsets[i]:offsets[i + 1]].tolist()
        i += 1
        active = set(tracker.state.get('tracks', {}))
        output = tracker(points, scan_scores)
        for batch, index in tracker.joint_eligible:
            kind = update_class(labels_by_id.get(batch, ()), labels[points[index]['PntID']])
            counts[f'eligible_{kind}'] += 1
        counts['proposals'] += len(tracker.joint_proposals)
        candidates = tracker.joint_proposals if shadow else tracker.joint_applied
        for p in candidates:
            errors, kind, helpful = audit(p, points, labels, labels_by_id)
            if shadow:
                controls.extend([p.gain] * errors)
            else:
                counts['reassignments'] += 1
                counts['swaps' if p.swap else 'transfers'] += 1
                counts['displaced_correct'] += errors
                counts['primary_correct' if kind == 1 else 'primary_other_target' if kind == 2
                       else 'primary_clutter' if kind == 3 else 'primary_unresolved'] += 1
                counts['label_confirmed_corrections'] += helpful
                received = {(item['BatchID'], item['PntID']) for item in output}
                changed = (p.rival, points[p.point]['PntID']) in received
                counts['published_reassignments'] += changed
                assert changed, (circle, p, tracker.last_matches.get(p.rival))
                decisions.append(dict(circle=circle, winner=p.winner, rival=p.rival, point=points[p.point]['PntID'],
                                      gain=p.gain, swap=p.swap, primary_kind=kind,
                                      displaced_correct=errors, label_confirmed=helpful))
        for item in output:
            truth = labels[item['PntID']]
            if truth >= 0:
                labels_by_id[item['BatchID']].append(truth)
        births.observe(points, labels, output, tracker, circle)
        diagnostic.observe(points, labels, output, tracker, active, circle)
        return output

    trace_path = out / f'{scene}.jsonl'
    with trace_path.open('w', encoding='utf-8') as trace:
        summary = replay_frames(frames, trace, handler)
    if shadow:
        trace_path.unlink()
    return dict(summary=summary, operations=dict(counts), controls=controls, decisions=decisions,
                identity=diagnostic.result()['counts'], birth_audit=births.result())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'prior', 'affinity', 'head', 'fusion', 'freeze', 'protocol', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.protocol.read_bytes())
    root = Path(__file__).resolve().parents[2]
    assert all(sha(root / name) == digest for name, digest in plan['sources'].items())
    role = 'work' if args.calibration is not None else 'development'
    assert sha(args.freeze) == plan['parent_freeze_sha256'] and sha(args.data) == plan[role]['data_sha256']
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    split = json.loads((root / 'splits/v3_tracking.json').read_bytes())
    names = plan[role]['scenes']
    allowed = split['fit64'] if role == 'development' else components.freeze['work_validation16']
    assert set(names) <= set(allowed) and not set(names) & set(split['sealed_excluded'])
    scene = names[args.index]
    cache_path = args.cache / f'{scene}.npz'
    manifest = json.loads((args.cache / 'manifest.json').read_bytes())
    assert sha(cache_path) == manifest['scenes'][scene]['cache_sha256']
    with open_scenes(args.data) as (_, opener), opener(scene) as source:
        frames = list(_scans(source))
    with np.load(cache_path) as cache:
        scores, offsets = cache['point_scores'], cache['point_offsets']
    assert len(offsets) == len(frames) + 1
    tracker = install(components.build('candidate'), 'combined')
    tracker.__class__ = JointCostTracker
    tracker.joint_cutoff = None
    digest = sha(args.protocol)
    if args.calibration is not None:
        calibration = json.loads(args.calibration.read_bytes())
        assert calibration['protocol_sha256'] == digest and calibration['scenes'] == plan['development']['scenes']
        tracker.joint_cutoff = calibration['cutoff']
    args.out.mkdir(parents=True, exist_ok=True)
    report = run(frames, scores, offsets, tracker, args.out, scene, shadow=role == 'development')
    save(args.out / f'{scene}.json', dict(report, scene=scene, role=role, protocol_sha256=digest,
                                          calibration_sha256=sha(args.calibration) if args.calibration else None))
    print(scene, role, report['operations'], flush=True)


if __name__ == '__main__':
    main()
