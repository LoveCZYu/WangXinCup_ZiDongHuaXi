"""Render the candidate-onset local result tables from archived aggregates."""
import argparse
import csv
from html import escape
import json
from pathlib import Path

STUDIES = {
    "直接改写状态 · 训练4": "training4-comparison",
    "仅在无候选时补救 · 训练4": "backup-training4-comparison",
    "补救与身份竞争 · 训练4": "mixed-training4-comparison",
    "补救与身份竞争 · 开发8": "mixed-development8-comparison",
    "冻结组合 · 工作16": "work16-comparison",
    "补救与身份竞争 · 工作16诊断": "work16-factorial-report",
}
MODES = {
    "baseline": "冻结原方案", "position": "改写位置", "velocity": "改写速度", "both": "改写位置和速度",
    "backup_motion": "运动补救", "backup_guarded": "带身份检查的补救", "backup": "带身份检查的补救",
    "column": "身份竞争", "combined": "组合", "guarded": "组合加身份分下限",
}
COLUMNS = ("本地总分", "零碎度", "连续子分", "点命中率 %", "已知杂点", "混轨少数点", "1 或 2 点额外 ID")


def read(path):
    return json.loads(path.read_bytes())


def results(report):
    for mode, row in (report.get("models") or report["aggregate"]).items():
        metrics = row["metrics"]
        yield [MODES[mode], *(f"{metrics[key]:.4f}" for key in
                            ("weighted_total", "fragmentation_score", "continuous_score", "point_recall")),
               metrics["clutter_outputs"],
               row["identity"]["mixed_minority_truth_points"] if "identity" in row else row["mixed_points"],
               row.get("short_ids_1or2", row.get("short_ids"))]


def table(headers, rows):
    def cells(tag, row):
        return "".join(f"<{tag}>{escape(str(value))}</{tag}>" for value in row)
    return ("<table><thead><tr>" + cells("th", headers) + "</tr></thead><tbody>" +
            "".join("<tr>" + cells("td", row) + "</tr>" for row in rows) + "</tbody></table>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    sections = []
    with (root / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["实验", "方案", *COLUMNS])
        for title, filename in STUDIES.items():
            report = read(root / (filename + ".json"))
            rows = list(results(report))
            writer.writerows([title, *row] for row in rows)
            failed = [key for key, passed in report.get("primary_checks", {}).items() if not passed]
            note = "<p>未通过的条件：" + escape("、".join(failed)) + "</p>" if failed else ""
            sections.append(f"<h2>{escape(title)}</h2>" + table(("方案", *COLUMNS), rows) + note)
    factorial = read(root / "work16-factorial-report.json")
    guard_path = root / "work16-guard-report.json"
    guard = read(guard_path) if guard_path.exists() else None
    if guard:
        rows = list(results({"aggregate": {"guarded": guard}}))
        with (root / "results.csv").open("a", newline="", encoding="utf-8-sig") as stream:
            csv.writer(stream).writerows(["身份分下限 · 工作16诊断", *row] for row in rows)
        sections.append("<h2>身份分下限 · 工作16诊断</h2>" + table(("方案", *COLUMNS), rows))
    per_scene = (
        [name, *(record["mixed"][mode] for mode in ("baseline", "column", "backup", "combined")),
         record["mixed"]["combined"] - record["mixed"]["baseline"], record["interaction"],
         *([guard["mixed_by_scene"][name]] if guard else [])]
        for name, record in factorial["scenes"].items()
    )
    sections.append("<h2>各工作场景混轨少数点</h2>" + table(
        ("场景", "冻结原方案", "身份竞争", "补救", "组合", "组合增量", "交互差值",
         *(("组合加身份分下限",) if guard else ())), per_scene))
    links = []
    for name in ("work16-011-baseline.html", "work16-011-combined.html",
                 "development8-152-baseline.html", "development8-152-combined.html", "first-loss-motion.png"):
        if (root / name).is_file():
            links.append(f'<li><a href="{name}">{escape(name)}</a></li>')
    if links:
        sections.append("<h2>回放</h2><ul>" + "".join(links) + "</ul>")
    prefix = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>B02 航迹续接实验结果</title>
<style>body{font:16px/1.6 system-ui,sans-serif;margin:2rem auto;padding:0 1rem;max-width:1100px;color:#dce5ef;background:#111923}
table{display:block;overflow:auto;border-collapse:collapse;background:#192638;font-variant-numeric:tabular-nums}
td,th{border-bottom:1px solid #435165;padding:.4rem .8rem;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}
a{color:#93c8ff}h2{margin-top:2rem}</style>
<h1>B02 航迹续接实验结果</h1>
<p>均为本地估分。工作16用于已暴露场景上的复核，组合的混轨保护条件未通过。</p>
<p><a href="results.csv">下载分数表</a></p>"""
    (root / "index.html").write_text(prefix + "".join(sections) + "</html>", encoding="utf-8")


if __name__ == "__main__":
    main()
