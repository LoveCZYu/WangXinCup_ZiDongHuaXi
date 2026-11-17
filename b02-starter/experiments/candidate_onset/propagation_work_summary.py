"""Score the original combination, truth-triggered diagnostic and causal policies."""
import argparse
from collections import Counter
import json
from pathlib import Path

from b02.scorecard import build_scorecard
from experiments.candidate_onset.propagation import sha


def read(path):
    return json.loads(path.read_bytes())


def load(root, pattern, expected_sha):
    result = {}
    for path in root.glob(pattern):
        row = read(path)
        assert row["protocol_sha256"] == expected_sha and row["scene"] not in result
        result[row["scene"]] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    for name in ("baseline", "oracle", "policy", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    names = read(args.policy)["scenes"]
    baseline = load(args.root, "work16-*/train_*.json", sha(args.baseline))
    oracle = load(args.root, "propagation-oracle-*/train_*.json", sha(args.oracle))
    policy = load(args.root, "propagation-policy-*/train_*.json", sha(args.policy))
    assert set(oracle) == set(policy) == set(names) <= set(baseline)
    original = {scene:dict(summary=baseline[scene]["variants"]["combined"],
                           identity=baseline[scene]["identity"]["combined"],
                           birth_audit=baseline[scene]["birth_audit"]["combined"],
                           shadow=baseline[scene]["shadow"]["combined"],operations={}) for scene in names}
    reference = dict(scenes={n:original[n]["summary"] for n in names})
    groups = {"combined": original, "truth_triggered_full": oracle}
    for mode in read(args.policy)["modes"]:
        groups[mode] = {n:policy[n]["variants"][mode] for n in names}
    result = {}
    for mode, rows in groups.items():
        card = build_scorecard(dict(scenes={n:row["summary"] for n,row in rows.items()}), mode, reference,"combined")
        mixed = {n:row["identity"]["mixed_minority_truth_points"] for n,row in rows.items()}
        counts = Counter()
        for row in rows.values():
            counts.update(row["operations"])
        result[mode] = dict(metrics=card["overall"]["values"], delta=card["overall"]["delta"],
                            mixed_points=sum(mixed.values()), mixed_by_scene=mixed, rollbacks=dict(counts),
                            scene_delta={n:row["delta"] for n,row in card["scenes"].items()},
                            short_ids=sum(row["birth_audit"]["extra_ids_1or2"] for row in rows.values()))
    rules = read(args.policy)["guards"]
    current = result["combined"]
    for mode in read(args.policy)["modes"]:
        row = result[mode]
        row["checks"] = {
            "mixed": row["mixed_points"] <= current["mixed_points"]*(1-rules["mixed_reduction_fraction"]),
            "point_recall": row["delta"]["point_recall"] >= -rules["point_recall_drop_pp"],
            "fragmentation": row["delta"]["fragmentation_score"] >= -rules["fragmentation_drop"],
            "continuity": row["delta"]["continuous_score"] >= -rules["continuous_drop"],
        }
    report = dict(scenes=names, models=result, artifact_shas={name:sha(getattr(args,name))
                 for name in ("baseline","oracle","policy")})
    args.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({mode:dict(total=row["metrics"]["weighted_total"],mixed=row["mixed_points"],
                     delta={key:row["delta"][key] for key in ("point_recall","fragmentation_score","continuous_score")},
                     rollbacks=row["rollbacks"],checks=row.get("checks"))
                     for mode,row in result.items()},indent=2))


if __name__ == "__main__":
    main()
