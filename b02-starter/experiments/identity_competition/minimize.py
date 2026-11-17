"""Delta-debug the actual one-step switch, keeping the old candidate present."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.identity_competition.repro import replay
from experiments.v2_association.run_stage1 import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fixture', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    manifest, packet, _, _ = replay(args.fixture)
    baseline = pickle.loads(packet['before'])
    old, new, target = manifest['expected_previous_id'], manifest['observed_new_id'], manifest['point_id']
    trials = 0

    def check(tracker, pairs):
        nonlocal trials
        trials += 1
        tracker = deepcopy(tracker)
        points, scores = map(list, zip(*pairs))
        result = tracker(points, scores)
        index = next(i for i, pt in enumerate(points) if pt['PntID'] == target)
        actual = next((pt['BatchID'] for pt in result if pt['PntID'] == target), None)
        edge = any(e['batch'] == old and index in e['indices'] for e in tracker.events)
        return actual == new and edge, result

    pairs = list(zip(packet['points'], packet['scores']))
    assert check(baseline, pairs)[0]
    original_tracks = len(baseline.state['tracks'])
    # First attempt the two IDs directly involved, preserving their real histories.
    minimal = deepcopy(baseline)
    keep = {old, new}
    minimal.state['tracks'] = {k: v for k, v in minimal.state['tracks'].items() if k in keep}
    for name in ('history', 'context_history', 'birth_scan', 'quality_hits', 'strong_hits', 'memories'):
        value = getattr(minimal, name)
        setattr(minimal, name, {k: v for k, v in value.items() if k in keep})
    minimal.retired = {}
    minimal.short_retired = {}
    if check(minimal, pairs)[0]:
        baseline = minimal
    size = max(1, len(pairs)//2)
    while size:
        changed = False
        for start in range(0, len(pairs), size):
            removed = pairs[start:start+size]
            if any(pt['PntID'] == target for pt, _ in removed):
                continue
            candidate = pairs[:start] + pairs[start+size:]
            if check(baseline, candidate)[0]:
                pairs = candidate
                changed = True
                break
        if not changed:
            size //= 2
    # Verify one-point irreducibility for the exact switch-and-old-edge symptom.
    for i, (point, _) in enumerate(pairs):
        if point['PntID'] != target:
            assert not check(baseline, pairs[:i]+pairs[i+1:])[0]
    _, observed = check(baseline, pairs)
    points, scores = map(list, zip(*pairs))
    raw = pickle.dumps({'before': pickle.dumps(baseline, protocol=5), 'points': points, 'scores': scores,
        'labels': {p['PntID']: packet['labels'][p['PntID']] for p in points}, 'observed_outputs': observed}, protocol=5)
    assert not args.out.exists()
    args.out.write_bytes(raw)
    manifest.update(fixture_sha256=hashlib.sha256(raw).hexdigest(), minimized_from_sha256=json.loads(
        args.fixture.with_suffix('.json').read_text())['fixture_sha256'], original_tracks=original_tracks,
        remaining_tracks=len(baseline.state['tracks']), original_points=len(packet['points']),
        remaining_points=len(points), trials=trials, point_irreducible_for_symptom=True)
    save(args.out.with_suffix('.json'), manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
