"""Whole-scene truth-triggered state rollback diagnostic on exposed work scenes."""
import argparse
from collections import Counter, defaultdict, deque
from copy import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from b02.replay import _scans, open_scenes, replay_frames
from experiments.candidate_onset.mixed import install
from experiments.candidate_onset.propagation import ASSOCIATION_FIELDS, anchor, edge_rows, restore, sha
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.robust_validation.oracles import shadow_identity
from experiments.v2_association.run_stage1 import Diagnostic, save


def capture_prefix(tracker, batches):
    prefix = SimpleNamespace(state={"tracks": {}}, retired={}, short_retired={})
    for field in ASSOCIATION_FIELDS:
        setattr(prefix, field, {})
    for batch in batches:
        track = tracker.state["tracks"].get(batch)
        if track is None:
            continue
        prefix.state["tracks"][batch] = copy(track)
        for field in ASSOCIATION_FIELDS:
            original = getattr(tracker, field)
            if batch not in original:
                continue
            saved = copy(original[batch])
            if field == "memories":
                saved.entries = copy(original[batch].entries)
            getattr(prefix, field)[batch] = saved
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cache", "prior", "affinity", "head", "fusion", "freeze", "protocol", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.protocol.read_bytes())
    assert sha(args.data)==plan["data_sha256"] and sha(args.freeze)==plan["parent_freeze_sha256"]
    root = Path(__file__).resolve().parents[2]
    for name,digest in plan["sources"].items():
        assert sha(root/name)==digest
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    scene = plan["scenes"][args.index]
    assert scene in components.freeze["work_validation16"]
    manifest = json.loads((args.cache/"manifest.json").read_bytes())
    cache_path = args.cache/(scene+".npz")
    assert sha(cache_path)==manifest["scenes"][scene]["cache_sha256"]
    with open_scenes(args.data) as (_, opener), opener(scene) as source:
        frames = list(_scans(source))
    with np.load(cache_path) as cache:
        raw_scores, offsets = cache["point_scores"], cache["point_offsets"]
    assert len(offsets)==len(frames)+1
    tracker = install(components.build("candidate"), "combined")
    histories = defaultdict(lambda: deque(maxlen=8))
    seen = set()
    counts = Counter()
    birth, diagnostic = BirthAudit(), Diagnostic()
    index = 0

    def run(points, _):
        nonlocal index
        circle, _, labels = frames[index]
        scores = raw_scores[offsets[index]:offsets[index+1]].tolist()
        index += 1
        prefix = capture_prefix(tracker, [batch for batch in tracker.state.get("tracks", {}) if anchor(histories.get(batch, ()))])
        before = set(tracker.state.get("tracks", {}))
        output = tracker(points, scores)
        by_batch = {point["BatchID"]:point["PntID"] for point in output}
        for (batch, point_index), truth, _, _, _ in edge_rows(tracker, points, scores, labels, histories):
            expected = anchor(histories[batch])
            if truth==expected or by_batch.get(batch)!=points[point_index]["PntID"]:
                continue
            key = batch, expected, truth
            if key in seen:
                continue
            seen.add(key)
            counts["eligible_wrong_events"] += 1
            counts["state_rollbacks"] += restore(tracker,prefix,batch,full=True)
        for point in output:
            truth = labels[point["PntID"]]
            if truth >= 0:
                histories[point["BatchID"]].append(truth)
        birth.observe(points,labels,output,tracker,circle)
        diagnostic.observe(points,labels,output,tracker,before,circle)
        return output

    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/(scene+".jsonl")).open("w",encoding="utf-8") as trace:
        summary = replay_frames(frames,trace,run)
    report = dict(scene=scene, protocol_sha256=sha(args.protocol), summary=summary,
                  identity=diagnostic.result()["counts"], birth_audit=birth.result(),
                  shadow=shadow_identity(summary), operations=dict(counts))
    save(args.out/(scene+".json"),report)
    print(scene,counts,report["identity"]["mixed_minority_truth_points"],flush=True)


if __name__=="__main__":
    main()
