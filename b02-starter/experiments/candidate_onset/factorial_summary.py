"""Compare the fixed four modes and locate the first affected mixed-track segment."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from b02.replay import _scans, open_scenes
from b02.scorecard import build_scorecard

MODES = ("baseline", "column", "backup", "combined")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(root, pattern, protocol):
    rows = {}
    for path in root.glob(pattern):
        row = read(path)
        name = row["scene"]
        assert name not in rows and row["protocol_sha256"] == protocol
        rows[name] = (path.parent, row)
    return rows


def score(rows, mode):
    return dict(scenes={scene: row["variants"][mode] for scene, row in rows.items()})


def trace_cases(root, folders, data, scene):
    with open_scenes(data) as (_, opener):
        with opener(scene) as source:
            labels_by_circle = {circle: labels for circle, _, labels in _scans(source)}
    results = {}
    for mode in MODES:
        histories = defaultdict(list)
        with (folders[scene] / mode / (scene + ".jsonl")).open(encoding="utf-8") as stream:
            for line in stream:
                frame = json.loads(line)
                circle = frame["circle"]
                for point in frame["tracks"]:
                    truth = labels_by_circle[circle][point["PntID"]]
                    if truth > 0:
                        histories[point["BatchID"]].append((circle, point["PntID"], truth))
        impacts = []
        pairs = Counter()
        for batch, history in histories.items():
            counts = Counter(event[2] for event in history)
            if len(counts) <= 1:
                continue
            main = max(counts, key=counts.get)
            for other, count in counts.items():
                if other != main:
                    pairs[tuple(sorted((main, other)))] += count
            start = next(event for event in history if event[2] != history[0][2])
            impacts.append(dict(batch=batch, truth_counts=dict(counts), first_truth=history[0][2],
                                first_different_truth=dict(zip(("circle", "point", "truth"), start)),
                                mixed_points=len(history)-counts[main]))
        results[mode] = dict(
            mixed_points=sum(item["mixed_points"] for item in impacts),
            top_mixed_tracks=sorted(impacts, key=lambda item: (-item["mixed_points"], item["batch"]))[:8],
            truth_pairs=[dict(truth=list(pair), mixed_points=count) for pair, count in pairs.most_common(10)],
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.protocol)
    original = args.root / "work16-comparison.json"
    assert digest(original) == plan["original_report_sha256"]
    assert digest(args.data) == plan["data_sha256"]
    pairs = load_rows(args.root, "work16-*/train_*.json", plan["parent_work_protocol_sha256"])
    factorial = load_rows(args.root, "factorial-*/train_*.json", digest(args.protocol))
    assert set(pairs) == set(factorial) == set(plan["work16"])
    rows, folders = {}, {}
    for scene in plan["work16"]:
        folder, extra = factorial[scene]
        _, previous = pairs[scene]
        for mode in ("baseline", "combined"):
            if mode in extra["variants"]:
                for field in ("identity", "birth_audit", "shadow", "operations"):
                    assert extra[field][mode] == previous[field][mode], (scene, mode, field)
                a, b = (dict(item["variants"][mode]) for item in (extra, previous))
                for key in ("p95_handler_ms", "max_handler_ms"):
                    a.pop(key, None)
                    b.pop(key, None)
                assert a == b, (scene, mode)
        rows[scene] = {field: previous[field] | extra[field] for field in
                       ("variants", "identity", "birth_audit", "shadow", "operations")}
        folders[scene] = folder

    scenes = {}
    for scene, row in rows.items():
        mixed = {mode: row["identity"][mode]["mixed_minority_truth_points"] for mode in MODES}
        scenes[scene] = dict(mixed=mixed,
                            interaction=mixed["combined"]-mixed["backup"]-mixed["column"]+mixed["baseline"])
    aggregate = {}
    for mode in MODES:
        result = build_scorecard(score(rows, mode), mode, score(rows, "baseline"), "baseline")
        aggregate[mode] = dict(metrics=result["overall"]["values"],
                               mixed_points=sum(record["mixed"][mode] for record in scenes.values()),
                               short_ids=sum(row["birth_audit"][mode]["extra_ids_1or2"] for row in rows.values()))
    cases = {scene: trace_cases(args.root, folders, args.data, scene) for scene in plan["case_scenes"]}
    for scene, modes in cases.items():
        for mode, case in modes.items():
            assert case["mixed_points"] == scenes[scene]["mixed"][mode], (scene, mode)
    report = dict(protocol_sha256=digest(args.protocol), original_gate=read(original)["primary_checks"],
                  aggregate=aggregate, scenes=scenes, cases=cases)
    (args.root / "work16-factorial-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({mode: dict(total=row["metrics"]["weighted_total"], mixed=row["mixed_points"])
                      for mode, row in aggregate.items()}, indent=2))
    print(json.dumps({scene: scenes[scene] for scene in plan["case_scenes"]}, indent=2))


if __name__ == "__main__":
    main()
