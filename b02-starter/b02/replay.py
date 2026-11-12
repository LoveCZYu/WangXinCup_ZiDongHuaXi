"""Replay published CSV scenes through the same circle handler as the TCP client."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from io import TextIOWrapper
from itertools import groupby
import json
from pathlib import Path, PurePosixPath
from struct import Struct
from time import perf_counter
from zipfile import ZipFile

from .scorecard import build_scorecard, write_scorecard

INPUT_FIELDS = (
    "Time", "Azimuth", "Elevation", "Distance", "Amp", "SNR", "CircleID", "PntID",
)
CSV_FIELDS = (
    "Pt_RadarTime", "Pt_Azi", "Pt_El", "Pt_Distance", "Pt_Amp", "Pt_SNR", "Pt_Circle", "Pt_PtID",
)
WIRE_POINT = Struct("<IfffffII")


def _uint(value):
    numeric = float(value)
    if not numeric.is_integer() or not 0 <= numeric <= 0xFFFFFFFF:
        raise ValueError(f"不是合法的 uint32 字段：{value!r}")
    return int(numeric)


def _point(row):
    values = [(_uint if i in (0, 6, 7) else float)(row[key]) for i, key in enumerate(CSV_FIELDS)]
    return dict(zip(INPUT_FIELDS, WIRE_POINT.unpack(WIRE_POINT.pack(*values))))


def _scans(source):
    rows = csv.DictReader(source)
    if not set((*CSV_FIELDS, "Batch_ID")).issubset(rows.fieldnames or ()):
        raise ValueError("CSV 缺少训练数据字段，请检查表头")
    previous = -1
    for circle, group in groupby(rows, key=lambda row: _uint(row["Pt_Circle"])):
        if circle <= previous:
            raise ValueError("CSV 的 Pt_Circle 必须按圈号非递减排列")
        for gap in range(previous + 1, circle):
            yield gap, [], {}
        points = []
        labels = {}
        for row in group:
            point = _point(row)
            pnt_id = point["PntID"]
            if pnt_id in labels:
                raise ValueError(f"圈 {circle} 存在重复点号 {pnt_id}")
            label = float(row["Batch_ID"])
            if not label.is_integer() or label < -1:
                raise ValueError(f"无效 Batch_ID：{label}")
            labels[pnt_id] = int(label)
            points.append(point)
        yield circle, points, labels
        previous = circle


def _identity(point):
    return tuple(point[k] for k in ("Time", "CircleID", "PntID", "Azimuth", "Elevation", "Distance"))


def _checked_outputs(outputs, originals):
    if not isinstance(outputs, list):
        raise ValueError("算法必须返回航迹列表，空结果返回 []")
    seen = set()
    for output in outputs:
        identity = _identity(output)
        if identity not in originals or identity in seen:
            raise ValueError("每条输出必须对应本圈唯一原始点")
        batch = output["BatchID"]
        if type(batch) is not int or not 0 < batch <= 0xFFFFFFFF:
            raise ValueError("输出 BatchID 必须为正 uint32")
        seen.add(identity)


def replay_scene(source, trace, handler):
    return replay_frames(_scans(source), trace, handler)


def replay_frames(frames, trace, handler):
    counts = Counter(circles=0, empty_circles=0, points=0, outputs=0, truth_points=0,
                     matched_target_points=0, clutter_outputs=0, unknown_outputs=0)
    truth = defaultdict(list)
    state = {}
    durations = []
    for circle, points, labels in frames:
        originals = {_identity(point) for point in points}
        start = perf_counter()
        outputs = handler(points, state)
        duration = (perf_counter() - start) * 1000
        _checked_outputs(outputs, originals)
        durations.append(duration)
        by_point = {p["PntID"]: p["BatchID"] for p in outputs}
        counts.update(circles=1, empty_circles=int(not labels), points=len(labels), outputs=len(outputs))
        for pnt_id, label in labels.items():
            batch = by_point.get(pnt_id)
            if label > 0:
                truth[label].append((circle, batch))
                counts["truth_points"] += 1
                counts["matched_target_points"] += batch is not None
            elif batch is not None:
                counts["clutter_outputs" if label == 0 else "unknown_outputs"] += 1
        trace.write(json.dumps({"circle": circle, "milliseconds": round(duration, 4), "tracks": outputs}) + "\n")

    if not durations:
        raise ValueError("CSV 没有点迹")
    per_target = []
    for label, observations in sorted(truth.items()):
        matched = [(circle, batch) for circle, batch in observations if batch is not None]
        assigned = Counter(batch for _, batch in matched)
        per_target.append({
            "truth_id": label, "truth_points": len(observations), "matched_points": len(matched),
            "output_ids": dict(sorted(assigned.items())),
            "first_output_delay_circles": matched[0][0] - observations[0][0] if matched else None,
            "most_matched_points_with_one_id": max(assigned.values(), default=0),
        })
    n = counts["truth_points"]
    times = sorted(durations)
    return {
        **counts, "target_point_recall": counts["matched_target_points"] / n if n else None,
        "truth_tracks": len(truth), "matched_truth_tracks": sum(bool(t["matched_points"]) for t in per_target),
        "split_truth_tracks": sum(len(t["output_ids"]) > 1 for t in per_target),
        "p95_handler_ms": round(times[(95 * len(times) - 1) // 100], 4),
        "max_handler_ms": round(times[-1], 4), "per_target": per_target,
    }


@contextmanager
def open_scenes(path):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            index = {PurePosixPath(name).stem: name for name in members}
            if len(index) != len(members):
                raise ValueError("训练 ZIP 包含重复的场景名")

            def open_scene(scene):
                return TextIOWrapper(archive.open(index[scene]), encoding="utf-8-sig", newline="")

            yield sorted(index), open_scene
    elif path.is_dir():
        index = {p.stem: p for p in sorted(path.glob("*.csv"))}
        yield list(index), lambda scene: index[scene].open(encoding="utf-8-sig", newline="")
    elif path.suffix.lower() == ".csv" and path.is_file():
        yield [path.stem], lambda _: path.open(encoding="utf-8-sig", newline="")
    else:
        raise ValueError(f"请提供训练 ZIP、CSV 或包含 CSV 的目录：{path}")


def replay(path, out, handler, config, scenes=None):
    with open_scenes(path) as (available, open_scene):
        selected = available if scenes is None else scenes
        missing = sorted(set(selected) - set(available))
        if not selected or missing:
            raise ValueError(f"没有可运行场景，或场景不在数据中：{', '.join(missing)}")
        out.mkdir(parents=True)
        summary = {"kind": "local_diagnostics", "config": config, "scenes": {}}
        for scene in selected:
            with open_scene(scene) as source, (out / f"{scene}.jsonl").open("w", encoding="utf-8") as trace:
                result = replay_scene(source, trace, handler)
            summary["scenes"][scene] = result
            print(f"{scene}: {result['circles']} 圈，{result['outputs']} 个输出点，"
                  f"目标点命中 {result['matched_target_points']}/{result['truth_points']}，"
                  f"已知杂波输出 {result['clutter_outputs']}", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_scorecard(out, build_scorecard(summary, out.name))
