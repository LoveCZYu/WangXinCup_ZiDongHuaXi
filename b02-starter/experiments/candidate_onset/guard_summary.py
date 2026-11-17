"""Measure the single fixed affinity guard on the exposed work16."""
import argparse
from collections import Counter
import json
from pathlib import Path

from b02.scorecard import build_scorecard, write_scorecard
from experiments.candidate_onset.factorial_summary import digest, load_rows, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.protocol)
    original = args.root / "work16-comparison.json"
    assert digest(original) == plan["original_report_sha256"]
    baseline_report = read(original)
    baseline = load_rows(args.root, "work16-*/train_*.json", plan["parent_work_protocol_sha256"])
    guarded = load_rows(args.root, "guard-*/train_*.json", digest(args.protocol))
    names = plan["work16"]
    assert set(baseline) == set(guarded) == set(names)
    summary = dict(kind="local_diagnostics", config=dict(mode="guarded", role="exposed_work16"),
                   scenes={scene: guarded[scene][1]["variants"]["guarded"] for scene in names})
    reference = dict(scenes={scene: baseline[scene][1]["variants"]["baseline"] for scene in names})
    card = build_scorecard(summary, "guarded", reference, "baseline")
    guard_metrics = card["overall"]["values"]
    base_metrics = baseline_report["models"]["baseline"]["metrics"]
    rules = plan["guards"]
    checks = {}
    for key, rule in (("weighted_total", "weighted_gain_min"), ("fragmentation_score", "fragmentation_gain_min")):
        checks[key] = guard_metrics[key] >= base_metrics[key] + rules[rule]
    for key, rule in (("continuous_score", "continuous_drop_max"), ("point_recall", "point_drop_pp_max"),
                      ("track_recall", "track_drop_pp_max"), ("timeliness_score", "timeliness_drop_max")):
        checks[key] = guard_metrics[key] >= base_metrics[key] - rules[rule]
    checks["clutter_outputs"] = guard_metrics["clutter_outputs"] <= base_metrics["clutter_outputs"] * rules["clutter_ratio_max"]
    mixed_by_scene = {}
    operations, shadow = Counter(), Counter()
    short_ids = 0
    for scene in names:
        row = guarded[scene][1]
        mixed_by_scene[scene] = row["identity"]["guarded"]["mixed_minority_truth_points"]
        operations.update(row["operations"]["guarded"])
        shadow.update({name: value for name, value in row["shadow"]["guarded"].items() if isinstance(value, int)})
        short_ids += row["birth_audit"]["guarded"]["extra_ids_1or2"]
    mixed = sum(mixed_by_scene.values())
    checks["mixed"] = mixed <= baseline_report["models"]["baseline"]["identity"]["mixed_minority_truth_points"] * rules["mixed_ratio_max"]
    strict_recall = 100 * shadow["one_to_one_correct_points"] / shadow["truth_points"]
    checks["strict_recall"] = strict_recall >= baseline_report["models"]["baseline"]["strict_one_to_one_recall"] - rules["strict_drop_pp_max"]
    scene_deltas = {scene: card["scenes"][scene]["delta"] for scene in names}
    report = dict(protocol_sha256=digest(args.protocol), metrics=guard_metrics, delta=card["overall"]["delta"],
                  mixed_points=mixed, mixed_by_scene=mixed_by_scene, short_ids=short_ids,
                  strict_recall=strict_recall, operations=dict(operations), scene_deltas=scene_deltas,
                  descriptive_checks=checks, independent_validation=False)
    folder = args.root / "work16-guard"
    folder.mkdir(exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.root / "work16-guard-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_scorecard(folder, card)
    print(json.dumps({key: report[key] for key in ("metrics", "mixed_points", "operations", "descriptive_checks")}, indent=2))


if __name__ == "__main__":
    main()
