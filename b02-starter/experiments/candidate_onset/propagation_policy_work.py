"""Replay causal motion/full rollback with the same fixed screen on work scenes."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import numpy as np
import torch

from b02.replay import _scans, open_scenes, replay_frames
from experiments.candidate_onset.mixed import install
from experiments.candidate_onset.propagation import anchor, sha
from experiments.candidate_onset.propagation_policy import Quarantine, Screen
from experiments.prebirth.replay import BirthAudit
from experiments.robust_validation.frozen import Components
from experiments.robust_validation.oracles import shadow_identity
from experiments.v2_association.run_stage1 import Diagnostic, save


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
    for name, digest in plan["sources"].items():
        assert sha(root/name)==digest
    weights = args.protocol.with_name("screen-model.json")
    assert sha(weights)==plan["model_sha256"]
    screen = Screen(weights)
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    scene = plan["scenes"][args.index]
    assert scene in components.freeze["work_validation16"]
    cache_path = args.cache/(scene+".npz")
    manifest = json.loads((args.cache/"manifest.json").read_bytes())
    assert sha(cache_path)==manifest["scenes"][scene]["cache_sha256"]
    with open_scenes(args.data) as (_, opener), opener(scene) as source:
        frames = list(_scans(source))
    with np.load(cache_path) as cache:
        raw_scores, offsets = cache["point_scores"], cache["point_offsets"]
    assert len(offsets)==len(frames)+1
    variants = {}
    for mode in plan["modes"]:
        tracker = install(components.build("candidate"), "combined")
        policy = Quarantine(tracker, screen, full=mode=="learned_full")
        histories = defaultdict(lambda: deque(maxlen=8))
        counts = Counter()
        birth, diagnostic = BirthAudit(), Diagnostic()
        index = 0

        def run(points, _):
            nonlocal index
            circle, _, labels = frames[index]
            scores = raw_scores[offsets[index]:offsets[index+1]].tolist()
            index += 1
            before = set(tracker.state.get("tracks", {}))
            output = policy(points,scores)
            for batch,point_index in policy.applied:
                expected = anchor(histories.get(batch, ()))
                truth = labels[points[point_index]["PntID"]]
                if expected is None or truth < 0:
                    counts["ambiguous"] += 1
                elif truth == expected:
                    counts["correct_update"] += 1
                else:
                    counts["wrong_update"] += 1
            for point in output:
                truth = labels[point["PntID"]]
                if truth >= 0:
                    histories[point["BatchID"]].append(truth)
            birth.observe(points,labels,output,tracker,circle)
            diagnostic.observe(points,labels,output,tracker,before,circle)
            return output

        folder = args.out/mode
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/(scene+".jsonl")).open("w",encoding="utf-8") as trace:
            summary = replay_frames(frames,trace,run)
        assert sum(counts.values()) == policy.counters["rollbacks"]
        variants[mode] = dict(summary=summary, identity=diagnostic.result()["counts"],
                              birth_audit=birth.result(),shadow=shadow_identity(summary),operations=dict(counts))
        print(scene, mode, counts, variants[mode]["identity"]["mixed_minority_truth_points"],flush=True)
    save(args.out/(scene+".json"),dict(scene=scene, protocol_sha256=sha(args.protocol), variants=variants))


if __name__=="__main__":
    main()
