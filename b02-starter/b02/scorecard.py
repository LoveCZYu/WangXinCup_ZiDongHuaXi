"""Summarize labelled replay runs using explicitly local diagnostics."""

import csv
import json
from math import ceil
from statistics import mean, median


# Each row belongs to a contest metric. Several diagnostics can describe one metric.
METRICS = (
    ("weighted_total", "加权总分", "本地估算总分（非官网成绩）", 100, "分"),
    ("point_recall", "空情掌握率", "目标点命中率", 30, "%"),
    ("track_recall", "空情掌握率", "逐航迹平均命中率", 30, "%"),
    ("clutter_outputs", "虚警率", "已知杂波输出", 10, "点"),
    ("clutter_score", "虚警率", "虚警子分估计 · 仅已知杂波", 10, "分"),
    ("split_tracks", "航迹零碎度", "使用多个输出 ID 的真值航迹", 20, "条"),
    ("fragmentation_score", "航迹零碎度", "零碎度子分估计", 20, "分"),
    ("timeliness_score", "起批及时性", "及时性子分估计 · 5 秒/圈", 20, "分"),
    ("delay_median", "起批及时性", "首次输出延迟 · 中位数", 20, "圈"),
    ("delay_p90", "起批及时性", "首次输出延迟 · P90", 20, "圈"),
    ("unmatched_tracks", "起批及时性", "始终未命中的真值航迹", 20, "条"),
    ("dominant_id_coverage", "最大连续跟踪时长", "主输出 ID 覆盖率 · 逐航迹平均", 20, "%"),
    ("continuous_score", "最大连续跟踪时长", "连续跟踪子分估计 · 同 ID 跨漏圈累计", 20, "分"),
    ("unknown_outputs", "补充诊断", "属性未确定点输出", None, "点"),
)

SCORE_WEIGHTS = {"track_recall": .3, "clutter_score": .1,
                 "fragmentation_score": .2, "timeliness_score": .2, "continuous_score": .2}
SCORE_POLICY = {
    "version": "local-estimate-v1",
    "label": "本地估算，非官网成绩；只适合相同数据、相同口径的版本对照。",
    "weights": SCORE_WEIGHTS,
    "assumptions": [
        "掌握率为逐真值航迹命中比例的平均值，不是所有点的总命中率。",
        "标签 -1 不计入目标或杂波；虚警子分只统计已知杂波输出。",
        "零碎度暂以至少命中一点作为掌握成功，成功航迹取 100/N 的平均值；无成功航迹记 0。官方成功判据尚未明确。",
        "及时性暂按 5 秒/圈，用首次命中圈减去真值首圈加 2 圈；延迟截断到 0–50 秒。始终未命中记 0，少于三点的真值暂不纳入该子分。",
        "及时性使用首次匹配点，不追溯匹配输出 ID 更早的杂波起点；第三圈与第三个观测点在漏圈时也可能不同，此处采用第三圈。",
        "连续跟踪按同一输出 ID 的匹配点总数除以真值点数减 2，上限 1；同 ID 跨漏圈累计，少于三点的真值不纳入该子分。",
        "整批回放合并各场景的真值航迹和点数后计算，不对场景分数简单平均。任一子分无法定义时总分留空，不重分配权重。",
    ],
}


def load_summary(run):
    path = run / "summary.json"
    if not path.is_file():
        raise ValueError(f"缺少回放统计文件：{path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("kind") != "local_diagnostics" or not summary.get("scenes"):
        raise ValueError(f"该文件不包含可用的训练数据回放统计：{path}")
    return summary


def _scope(scenes):
    targets = [target for scene in scenes for target in scene["per_target"]]
    truth_points = sum(scene["truth_points"] for scene in scenes)
    hits = sum(scene["matched_target_points"] for scene in scenes)
    clutter = sum(scene["clutter_outputs"] for scene in scenes)
    delays = sorted(t["first_output_delay_circles"] for t in targets
                    if t["first_output_delay_circles"] is not None)
    # The task PDF excludes the first two observations from the denominator.
    # Each output ID is a single track even if its matched observations have gaps.
    continuous = [min(1, t["most_matched_points_with_one_id"] / (t["truth_points"] - 2))
                  for t in targets if t["truth_points"] > 2]
    fragmented = [1 / len(t["output_ids"]) for t in targets if t["output_ids"]]
    timely = [0 if t["first_output_delay_circles"] is None else
              1 - min(50, max(0, (t["first_output_delay_circles"] - 2) * 5)) / 50
              for t in targets if t["truth_points"] > 2]
    result = {
        "scene_count": len(scenes),
        "truth_points": truth_points,
        "truth_tracks": len(targets),
        "values": {
            "point_recall": 100 * hits / truth_points if truth_points else None,
            "track_recall": 100 * mean(t["matched_points"] / t["truth_points"] for t in targets) if targets else None,
            "clutter_outputs": clutter,
            "clutter_score": 100 * max(0, 1 - clutter / truth_points) if truth_points else None,
            "split_tracks": sum(len(t["output_ids"]) > 1 for t in targets),
            "fragmentation_score": 100 * mean(fragmented) if fragmented else (0 if targets else None),
            "timeliness_score": 100 * mean(timely) if timely else None,
            "delay_median": median(delays) if delays else None,
            "delay_p90": delays[ceil(0.9 * len(delays)) - 1] if delays else None,
            "unmatched_tracks": len(targets) - len(delays),
            "dominant_id_coverage": 100 * mean(t["most_matched_points_with_one_id"] / t["truth_points"]
                                               for t in targets) if targets else None,
            "continuous_score": 100 * mean(continuous) if continuous else None,
            "unknown_outputs": sum(scene["unknown_outputs"] for scene in scenes),
        },
    }
    values = result["values"]
    values["weighted_total"] = (sum(values[key] * weight for key, weight in SCORE_WEIGHTS.items())
                                if all(values[key] is not None for key in SCORE_WEIGHTS) else None)
    return result


def _scene_signature(scene):
    return (scene["circles"], scene["points"], scene["truth_points"],
            sorted((t["truth_id"], t["truth_points"]) for t in scene["per_target"]))


def build_scorecard(summary, run_name, baseline=None, baseline_name=None):
    scenes = summary["scenes"]
    reference = baseline["scenes"] if baseline is not None else None
    if reference is not None:
        for name, scene in scenes.items():
            if name not in reference:
                raise ValueError(f"基线回放缺少场景 {name}，无法对照")
            if _scene_signature(scene) != _scene_signature(reference[name]):
                raise ValueError(f"场景 {name} 的圈数、点数或真值覆盖不同，无法对照")

    def scope(names):
        result = _scope([scenes[name] for name in names])
        if reference is not None:
            values = _scope([reference[name] for name in names])["values"]
            result["baseline"] = values
            result["delta"] = {}
            for key, value in result["values"].items():
                old = values[key]
                result["delta"][key] = value - old if value is not None and old is not None else None
        return result

    return {
        "kind": "local_diagnostics_scorecard",
        "run": run_name,
        "baseline_run": baseline_name,
        "scoring_policy": SCORE_POLICY,
        "metrics": [{"key": key, "group": group, "label": label, "weight": weight, "unit": unit}
                    for key, group, label, weight, unit in METRICS],
        "overall": scope(list(scenes)),
        "scenes": {name: scope([name]) for name in scenes},
    }


def scorecard_for_run(run, baseline=None):
    return build_scorecard(load_summary(run), run.name,
                           load_summary(baseline) if baseline is not None else None,
                           baseline.name if baseline is not None else None)


def write_scorecard(run, card):
    (run / "scorecard.json").write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run / "scorecard-policy.txt").write_text(
        SCORE_POLICY["version"] + "\n" + SCORE_POLICY["label"] + "\n"
        + "\n".join(SCORE_POLICY["assumptions"]) + "\n", encoding="utf-8")
    with (run / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["范围", "本地估算总分（非官网）", "掌握率子分", "虚警子分", "零碎度子分",
                         "及时性子分", "连续跟踪子分", "基线总分", "总分变化", "计算口径"])
        for name, scope in [("整批回放", card["overall"]), *card["scenes"].items()]:
            writer.writerow([name, scope["values"]["weighted_total"],
                             *(scope["values"][key] for key in SCORE_WEIGHTS),
                             scope.get("baseline", {}).get("weighted_total"),
                             scope.get("delta", {}).get("weighted_total"), SCORE_POLICY["version"]])
    with (run / "scorecard.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["范围", "场景数", "真值点数", "真值航迹数", "赛题指标", "指标权重(%)", "本地诊断",
                         "单位", "当前版本", "当前值", "基线版本", "基线值", "变化（比例为百分点）"])
        for name, scope in [("整批回放", card["overall"]), *card["scenes"].items()]:
            for key, group, label, weight, unit in METRICS:
                writer.writerow([name, scope["scene_count"], scope["truth_points"], scope["truth_tracks"],
                                 group, weight, label, unit, card["run"], scope["values"][key],
                                 card["baseline_run"], scope.get("baseline", {}).get(key),
                                 scope.get("delta", {}).get(key)])
