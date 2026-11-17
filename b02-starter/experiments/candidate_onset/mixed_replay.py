"""Replay the onset of the most increased mixed truth pair from one shared prefix."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from b02.replay import _scans, open_scenes
from experiments.candidate_onset.evidence_guard import EvidenceGuard
from experiments.candidate_onset.mixed import MODES, install
from experiments.candidate_onset.observe import build
from experiments.id_recovery.affinity import pack
from experiments.robust_validation.frozen import Components


def settings(tracker, mode):
    tracker.backup_mode = "backup_guarded" if mode in ("backup", "combined") else "baseline"
    tracker.column_evidence.mode = "identity" if mode in ("column", "combined") else "baseline"


def case_choice(modes):
    baseline = {tuple(row["truth"]): row["mixed_points"] for row in modes["baseline"]["truth_pairs"]}
    candidates = modes["combined"]["top_mixed_tracks"]
    def impact(row):
        first, wrong = row["first_truth"], row["first_different_truth"]["truth"]
        return row["mixed_points"]-baseline.get(tuple(sorted((first, wrong))), 0)
    return max(candidates, key=impact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cache", "prior", "affinity", "head", "fusion", "freeze", "report", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = json.loads(args.report.read_bytes())
    components = Components(args.prior, args.affinity, args.head, args.fusion, args.freeze)
    results = {}
    with open_scenes(args.data) as (_, opener):
        for scene, modes in report["cases"].items():
            assert scene in components.freeze["work_validation16"]
            case = case_choice(modes)
            onset = case["first_different_truth"]
            batch = case["batch"]
            with np.load(args.cache / (scene + ".npz")) as stored:
                scores = stored["point_scores"]
                offsets = stored["point_offsets"]
            tracker = install(build(components), "combined")
            prefix = Counter()
            trace = []
            with opener(scene) as source:
                for index, (circle, points, labels) in enumerate(_scans(source)):
                    values = scores[offsets[index]:offsets[index+1]].tolist()
                    if circle != onset["circle"]:
                        output = tracker(points, values)
                        prefix.update(labels[point["PntID"]] for point in output if point["BatchID"] == batch)
                        continue
                    alternatives = {}
                    evidence = {}
                    for mode in (*MODES, "guarded"):
                        ablation = deepcopy(tracker)
                        if mode == "guarded":
                            ablation.model_scorer = EvidenceGuard(ablation.column_evidence.prior, ablation)
                        else:
                            settings(ablation, mode)
                        before = ablation.motion_counts["backup_recovered"]
                        output = ablation(points, values)
                        batch_points = [
                            dict(point=point["PntID"], truth=labels[point["PntID"]])
                            for point in output if point["BatchID"] == batch
                        ]
                        decision = getattr(ablation, "decisions", {})
                        alternatives[mode] = dict(batch_points=batch_points,
                            disputed_point_owner=next((point["BatchID"] for point in output
                                                       if point["PntID"] == onset["point"]), None),
                            backup_matches=ablation.motion_counts["backup_recovered"]-before,
                            continuation_transfers=[list(x) for x in decision if batch in x[1::2]])
                        point_index = next(i for i, point in enumerate(points)
                                           if point["PntID"] == onset["point"])
                        evidence[mode] = []
                        for event in ablation.events:
                            if point_index not in event["indices"]:
                                continue
                            edge = event["indices"].index(point_index)
                            affiliation = ablation.affinity.score(pack(
                                event["track_features"][None, :],
                                event["edge_features"][edge:edge+1],
                            ))
                            evidence[mode].append(dict(batch=event["batch"],
                                normalized_residual_sq=float(event["edge_features"][edge, 9]),
                                affinity=float(affiliation[0]),
                                base_cost=float(event["base_costs"][edge]),
                                final_cost=float(event["effective_costs"][edge]),
                                candidate_count=len(event["indices"])))
                    assert alternatives["combined"]["disputed_point_owner"] == batch
                    trace.append(dict(circle=circle, batch=batch, prefix_truth_counts=dict(prefix),
                                      onset=onset, alternatives=alternatives, evidence=evidence))
                    break
            assert trace, scene
            results[scene] = trace[0]
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
