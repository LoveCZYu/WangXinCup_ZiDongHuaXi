"""Collect hard edges and replay one-shot interventions from one shared prefix."""
import argparse
from collections import Counter, defaultdict, deque
from copy import deepcopy
import hashlib
import json
from math import log
from pathlib import Path

import numpy as np
import torch

from b02.algorithm import POSITION
from b02.replay import _scans, open_scenes
from experiments.candidate_onset.mixed import install
from experiments.id_recovery.affinity import pack
from experiments.robust_validation.frozen import Components

MIN_AFFINITY = log(9)
HORIZON = 4
MAX_CASES = 12
MODES = ("original", "veto", "restore_motion", "restore_state")
ASSOCIATION_FIELDS = ("history", "context_history", "quality_hits", "strong_hits", "memories")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor(history):
    return history[0] if len(history) >= 4 and history[0] > 0 and len(set(history)) == 1 else None


def bank(tracker, batch):
    active = tracker.state.get("tracks", {}).get(batch)
    if active is not None:
        return active, "active"
    for name in ("retired", "short_retired"):
        entry = getattr(tracker, name).get(batch)
        if entry is not None:
            return entry.track, name
    return None, "absent"


def edge_rows(tracker, points, scores, labels, histories):
    events = []
    lookup = {}
    for event in tracker.events:
        batch = event["batch"]
        if not event["indices"] or not anchor(histories.get(batch, ())):
            continue
        for edge, index in enumerate(event["indices"]):
            lookup[(batch, index)] = (event, edge)
    for batch, index in tracker.last_matches.items():
        key = (batch, index)
        truth = labels[points[index]["PntID"]]
        if key in lookup and truth > 0 and scores[index] >= .525:
            events.append((key, truth, *lookup[key]))
    if not events:
        return []
    scores_affinity = tracker.affinity.score(pack(
        np.stack([event["track_features"] for _, _, event, _ in events]),
        np.stack([event["edge_features"][edge] for _, _, event, edge in events]),
    ))
    return [(key, truth, float(affinity), event, edge) for (key, truth, event, edge), affinity
            in zip(events, scores_affinity, strict=True) if affinity >= MIN_AFFINITY]


def collect(frames, cached, make_tracker):
    tracker = make_tracker()
    histories = defaultdict(lambda: deque(maxlen=8))
    seen_errors = set()
    seen_controls = set()
    errors, controls = [], []
    counts = Counter()
    reference = []
    for index, (circle, points, labels) in enumerate(frames):
        scores = cached(index)
        output = tracker(points, scores)
        by_batch = {p["BatchID"]: p["PntID"] for p in output}
        reference.append(tuple((p["BatchID"], p["PntID"]) for p in output))
        for (batch, point_index), truth, affinity, event, edge in edge_rows(tracker, points, scores, labels, histories):
            expected = anchor(histories[batch])
            wrong = truth != expected
            counts["wrong_edges" if wrong else "correct_edges"] += 1
            key = (batch, expected, truth)
            seen, destination = (seen_errors, errors) if wrong else (seen_controls, controls)
            if key in seen:
                continue
            seen.add(key)
            counts["wrong_events" if wrong else "correct_events"] += 1
            sample = dict(index=index, circle=circle, batch=batch, point=points[point_index]["PntID"],
                          expected_truth=expected, assigned_truth=truth, affinity=affinity,
                          detector_score=float(scores[point_index]), history_labels=list(histories[batch]),
                          edge_features=event["edge_features"][edge].tolist(),
                          track_features=event["track_features"].tolist(),
                          motion_residual_sq=float(event["edge_features"][edge, 9]),
                          candidate_count=len(event["indices"]),
                          correct_edge_available=any(labels[points[i]["PntID"]]==expected for i in event["indices"]),
                          strong_output=by_batch.get(batch)==points[point_index]["PntID"],
                          trusted_memory_update=batch in tracker.trusted_updates,
                          route=[decision[0] for decision in tracker.decisions if decision[1]==batch])
            destination.append(sample)
        for point in output:
            truth = labels[point["PntID"]]
            if truth >= 0:
                histories[point["BatchID"]].append(truth)
    return dict(counts), errors, controls, reference


def restore(tracker, prefix, batch, *, full):
    old, location = bank(prefix, batch)
    active = tracker.state["tracks"].get(batch)
    if location != "active" or active is None:
        return False
    if full:
        tracker.state["tracks"][batch] = deepcopy(old)
        for field in ASSOCIATION_FIELDS:
            before, after = getattr(prefix, field), getattr(tracker, field)
            if batch in before:
                after[batch] = deepcopy(before[batch])
            else:
                after.pop(batch, None)
    else:
        # Advance the old constant-velocity prediction to the existing update time.
        # This keeps hit count and observation time fixed while removing the innovation.
        elapsed = active.last_scan - old.last_scan
        active.position = tuple(p + elapsed*v for p, v in zip(old.position, old.velocity))
        active.velocity = old.velocity
    return True


def veto_this_scan(tracker, batch, point_index):
    assign = tracker.assign

    def wrapper(*args, **kwargs):
        matched = assign(*args, **kwargs)
        if matched.get(batch) == point_index:
            matched.pop(batch)
        return matched

    # Preserve the original solver and the original occupied set.
    tracker.assign = wrapper


def measure(tracker, output, points, labels, batch, truth):
    selected = [labels[point["PntID"]] for point in output if point["BatchID"] == batch]
    index = tracker.last_matches.get(batch)
    matched_truth = labels[points[index]["PntID"]] if index is not None else None
    state, location = bank(tracker, batch)
    available = [p for p in points if labels[p["PntID"]] == truth]
    error = None
    if state is not None and available:
        elapsed = tracker.state["scan"]-state.last_scan
        prediction = np.asarray(state.position)+elapsed*np.asarray(state.velocity)
        error = float(min(np.linalg.norm(prediction-np.asarray([
            p[key]/scale for key, scale in zip(POSITION, tracker.config.gate)])) for p in available))
    return dict(published_truth=selected, internal_truth=matched_truth,
                correct_observed=bool(available), state_bank=location,
                position=list(state.position) if state else None, velocity=list(state.velocity) if state else None,
                last_scan=state.last_scan if state else None,
                trusted_memory_size=len(tracker.memories[batch].entries) if batch in tracker.memories else 0,
                residual=error)


def replay_event(prefix, event, frames, cached, reference):
    start, batch, expected = event["index"], event["batch"], event["expected_truth"]
    modes = {}
    for mode in MODES:
        tracker = deepcopy(prefix)
        records = []
        applied = True
        for index in range(start, min(len(frames), start + HORIZON + 1)):
            circle, points, labels = frames[index]
            if index == start and mode == "veto":
                point_index = next(i for i, point in enumerate(points) if point["PntID"] == event["point"])
                veto_this_scan(tracker, batch, point_index)
            output = tracker(points, cached(index))
            if index == start:
                if mode == "veto":
                    del tracker.assign
                    assert not any(point["BatchID"]==batch and point["PntID"]==event["point"] for point in output)
                else:
                    assert tuple((point["BatchID"], point["PntID"]) for point in output)==reference[index]
                if mode.startswith("restore"):
                    applied = restore(tracker, prefix, batch, full=mode=="restore_state")
            if mode == "original":
                assert tuple((point["BatchID"], point["PntID"]) for point in output)==reference[index]
            records.append(dict(delta=index-start, circle=circle,
                                **measure(tracker, output, points, labels, batch, expected)))
        modes[mode] = dict(applied=applied, frames=records)
    return modes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cache", "prior", "affinity", "head", "fusion", "freeze", "protocol", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--role", choices=("development", "work"), required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.protocol.read_bytes())
    assert sha(Path(__file__)) == plan["source_sha256"]
    assert sha(args.data) == plan[args.role]["data_sha256"]
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    def make_tracker():
        return install(components.build("candidate"), "combined")
    split = json.loads((Path(__file__).resolve().parents[2] / "splits/v3_tracking.json").read_bytes())
    names = plan[args.role]["scenes"]
    assert not set(names) & set(split["sealed_excluded"])
    allowed = split["fit64"] if args.role=="development" else components.freeze["work_validation16"]
    assert set(names) <= set(allowed)
    args.out.mkdir(parents=True, exist_ok=True)
    with open_scenes(args.data) as (_, opener):
        for scene in names[args.index::args.shards]:
            with opener(scene) as source:
                frames = list(_scans(source))
            with np.load(args.cache / (scene + ".npz")) as cache:
                raw_scores, offsets = cache["point_scores"], cache["point_offsets"]
            assert len(offsets) == len(frames)+1
            def cached(index):
                return raw_scores[offsets[index]:offsets[index+1]].tolist()
            counts, errors, controls, reference = collect(frames, cached, make_tracker)
            if len(controls) > 3*MAX_CASES:
                controls = [controls[i] for i in np.linspace(0, len(controls)-1, 3*MAX_CASES, dtype=int)]
            eligible = [row for row in errors if row["strong_output"] and row["index"]+HORIZON<len(frames)]
            selected = [eligible[i] for i in np.linspace(0, len(eligible)-1, min(MAX_CASES, len(eligible)), dtype=int)] if eligible else []
            by_index = defaultdict(list)
            for sample in selected:
                by_index[sample["index"]].append(sample)
            tracker = make_tracker()
            for index in range(max(by_index, default=-1)+1):
                for sample in by_index[index]:
                    sample["replay"] = replay_event(tracker, sample, frames, cached, reference)
                output = tracker(frames[index][1], cached(index))
                assert tuple((point["BatchID"], point["PntID"]) for point in output)==reference[index]
            report = dict(scene=scene, role=args.role, counts=counts, errors=errors,
                          correct_controls=controls, replayed=len(selected), protocol_sha256=sha(args.protocol))
            (args.out / (scene + ".json")).write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")
            print(scene, counts, "eligible", len(eligible), "replayed", len(selected), flush=True)


if __name__ == "__main__":
    main()
