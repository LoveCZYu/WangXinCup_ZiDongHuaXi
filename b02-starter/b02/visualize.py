"""Export a self-contained browser viewer for one labelled replay scene."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from .replay import _checked_outputs, _identity, _scans, open_scenes
from .scorecard import build_scorecard, load_summary, write_scorecard


def _scene_data(source, trace, scene):
    scans = []
    truth = defaultdict(list)
    totals = {"points": 0, "truth": 0, "hits": 0, "clutter": 0, "unknown": 0}
    for (circle, points, labels), line in zip(_scans(source), trace, strict=True):
        frame = json.loads(line)
        if frame["circle"] != circle:
            raise ValueError(f"回放圈号与训练数据不一致，期望第 {circle} 圈")
        outputs = frame["tracks"]
        try:
            _checked_outputs(outputs, {_identity(point) for point in points})
        except ValueError as exc:
            raise ValueError(f"第 {circle} 圈的回放结果与输入点迹不一致：{exc}") from exc
        assigned = {track["PntID"]: track["BatchID"] for track in outputs}
        rows = []
        misses = clutter = unknown = hits = 0
        for point in points:
            point_id = point["PntID"]
            label = labels[point_id]
            batch = assigned.get(point_id, 0)
            rows.append([point_id, point["Time"], round(point["Azimuth"], 7),
                         round(point["Elevation"], 7), round(point["Distance"], 7),
                         label, batch])
            if label > 0:
                truth[label].append([circle, batch])
                hits += bool(batch)
                misses += not batch
            elif batch:
                clutter += label == 0
                unknown += label == -1
        totals["points"] += len(points)
        totals["truth"] += misses + hits
        totals["hits"] += hits
        totals["clutter"] += clutter
        totals["unknown"] += unknown
        scans.append({"circle": circle, "rows": rows, "hits": hits, "misses": misses,
                      "clutter": clutter, "unknown": unknown, "ms": frame["milliseconds"]})
    if not scans:
        raise ValueError("场景没有可显示的回放圈")

    targets = []
    for label, observations in sorted(truth.items()):
        matched = [(circle, batch) for circle, batch in observations if batch]
        switches = [after for before, after in zip(matched, matched[1:]) if before[1] != after[1]]
        targets.append({"id": label, "observations": observations,
                        "ids": sorted({batch for _, batch in matched}), "switches": switches,
                        "first": matched[0][0] if matched else None, "hits": len(matched)})
    return {"scene": scene, "scans": scans, "targets": targets, "totals": totals}


def _viewer_scorecard(payload, run, baseline, previous=None):
    targets = []
    for target in payload["targets"]:
        counts = Counter(batch for _, batch in target["observations"] if batch)
        targets.append({
            "truth_id": target["id"], "truth_points": len(target["observations"]),
            "matched_points": target["hits"], "output_ids": dict(counts),
            "first_output_delay_circles": (target["first"] - target["observations"][0][0]
                                            if target["first"] is not None else None),
            "most_matched_points_with_one_id": max(counts.values(), default=0),
        })
    totals = payload["totals"]
    scene = payload["scene"]
    local = {"circles": len(payload["scans"]), "points": totals["points"],
             "truth_points": totals["truth"], "matched_target_points": totals["hits"],
             "clutter_outputs": totals["clutter"], "unknown_outputs": totals["unknown"],
             "per_target": targets}
    if (run / "summary.json").is_file():
        summary = load_summary(run)
        recorded = summary["scenes"].get(scene, {})
        if any(recorded.get(key) != value for key, value in local.items() if key != "per_target"):
            raise ValueError(f"场景 {scene} 的逐圈结果与 summary.json 不一致")
        summary["scenes"][scene] = local
    else:
        summary = {"scenes": {scene: local}}
    reference_path = previous if previous is not None else baseline
    reference = load_summary(reference_path) if reference_path is not None else None
    card = build_scorecard(summary, run.name, reference, reference_path.name if reference is not None else None)
    if previous is not None:
        if baseline is None:
            raise ValueError("三版对照需要同时提供 --baseline 和 --previous")
        initial = build_scorecard(summary, run.name, load_summary(baseline), baseline.name)
        card["versions"] = [
            {"label": label, "run": path.name,
             "overall": source["overall"][key],
             "scenes": {name: scope[key] for name, scope in source["scenes"].items()}}
            for label, path, source, key in (
                ("基线", baseline, initial, "baseline"),
                ("上一版", previous, card, "baseline"),
                ("这一版", run, card, "values"),
            )
        ]
    return card


def export_viewer(data, run, scene, out, baseline=None, previous=None, replace=False):
    if out.suffix.lower() != ".html":
        raise ValueError("输出文件名需要以 .html 结尾")
    with open_scenes(data) as (available, open_scene):
        if scene not in available:
            raise ValueError(f"数据中没有场景 {scene}")
        trace_path = run / f"{scene}.jsonl"
        if not trace_path.is_file():
            raise ValueError(f"缺少 {trace_path}，请先回放该场景")
        with open_scene(scene) as source, trace_path.open(encoding="utf-8") as trace:
            payload = _scene_data(source, trace, scene)
    card = _viewer_scorecard(payload, run, baseline, previous)
    payload["scorecard"] = card
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w" if replace else "x", encoding="utf-8") as destination:
        destination.write(template.replace("__B02_PAYLOAD__", encoded.replace("<", r"\u003c")))
    if previous is None:
        write_scorecard(run, card)
    else:
        (run / "viewer-comparison.json").write_text(
            json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (run / "viewer-comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["范围", "指标", "单位", *(v["label"] for v in card["versions"])])
            for scope in ["整批回放", *card["scenes"]]:
                for metric in card["metrics"]:
                    values = [v["overall"] if scope == "整批回放" else v["scenes"][scope]
                              for v in card["versions"]]
                    writer.writerow([scope, metric["label"], metric["unit"],
                                     *(value[metric["key"]] for value in values)])
    print(f"已导出 {scene} 的 {len(payload['scans'])} 圈：{out.resolve()}")
