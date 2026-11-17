import argparse
from dataclasses import asdict
from functools import partial
import json
from math import isfinite
from pathlib import Path
import subprocess
import sys

from .algorithm import Config, process_circle
from .official import prepare, run_official
from .replay import replay
from .scorecard import scorecard_for_run, write_scorecard
from .visualize import export_viewer

ROOT = Path(__file__).resolve().parent.parent


def load_split(name):
    if name == 'holdout_unseen':
        policy=json.loads((ROOT/'splits/a_board.json').read_text(encoding='utf-8'))
        return policy['remaining_independent_holdout']
    return json.loads((ROOT/'splits/v1.json').read_text(encoding='utf-8'))[name]


def load_config(path):
    if path is None:
        return Config()
    config = Config(**json.loads(path.read_text(encoding="utf-8-sig")))
    if len(config.gate) != 3 or any(not isfinite(x) or x <= 0 for x in config.gate):
        raise ValueError("gate 需要三个有限正数")
    for key in ("confirm_hits", "max_missed", "max_tentative", "candidates_per_track"):
        value = getattr(config, key)
        if type(value) is not int or value < (0 if key == "max_missed" else 1):
            raise ValueError(f"{key} 必须为合法整数")
    if not 0 <= config.velocity_gain <= 1:
        raise ValueError("velocity_gain 必须在 0 到 1 之间")
    return config


def main():
    parser = argparse.ArgumentParser(description="B02 逐圈跟踪开发工具")
    commands = parser.add_subparsers(dest="command", required=True)
    local = commands.add_parser("replay", help="逐圈回放训练 CSV，并输出本地诊断")
    local.add_argument("data", type=Path, help="训练 ZIP、单个 CSV 或 CSV 目录")
    local.add_argument("--out", required=True, type=Path)
    local.add_argument("--config", type=Path)
    local.add_argument("--tracker", choices=("baseline", "point-scorer", "track-mt3"), default="baseline")
    local.add_argument("--checkpoint", type=Path, help="配套模型权重，点模型默认使用配置中的文件")
    local.add_argument("--profile", type=Path, help="点模型配置，默认 profiles/current.json")
    local.add_argument("--upstream", type=Path, help="prepare_upstream.py 生成的三维适配上游目录")
    local.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    selection = local.add_mutually_exclusive_group()
    selection.add_argument("--scene", action="append", help="场景名，不带 .csv。可多次传入")
    selection.add_argument("--split", choices=("train", "dev", "holdout", "holdout_unseen"),
                           help="holdout 是历史 30 场景；holdout_unseen 排除已做标签诊断的 CSV train_003 后为 29 场景")
    selection.add_argument("--all", action="store_true", help="明确回放数据源中的所有场景")

    viewer = commands.add_parser("visualize", help="导出可直接打开的离线场景查看器")
    viewer.add_argument("data", type=Path, help="训练 ZIP、单个 CSV 或 CSV 目录")
    viewer.add_argument("--run", required=True, type=Path, help="已有回放的输出目录")
    viewer.add_argument("--scene", required=True, help="场景名，不带 .csv")
    viewer.add_argument("--out", required=True, type=Path, help="新的 .html 文件路径")
    viewer.add_argument("--baseline", type=Path, help="最初简单基线的回放目录")
    viewer.add_argument("--previous", type=Path, help="上一版回放目录，与 --baseline 一起生成三版对照")
    viewer.add_argument("--replace", action="store_true", help="重新生成已有查看器文件")

    scorecard = commands.add_parser("scorecard", help="把本地结果表写入回放目录")
    scorecard.add_argument("run", type=Path, help="包含 summary.json 的回放目录")
    scorecard.add_argument("--baseline", type=Path, help="用于对照的基线回放目录")

    setup = commands.add_parser("prepare-official", help="从官方 ZIP 准备原版收发程序与样例")
    setup.add_argument("archive", type=Path)
    setup.add_argument("--demo-dir", type=Path, default=Path(".official"))

    official = commands.add_parser("official", help="经官方 TCP 收发程序运行")
    official.add_argument("--demo-dir", type=Path, default=Path(".official"))
    official.add_argument("--input-dir", type=Path)
    official.add_argument("--out", required=True, type=Path)
    official.add_argument("--port", type=int, default=9902)
    official.add_argument("--config", type=Path)
    official.add_argument("--original", action="store_true", help="运行官方首点示例以排查收发环境")
    official.add_argument("--tracker", choices=("baseline", "point-scorer"), default="baseline")
    official.add_argument("--profile", type=Path, help="点模型配置，默认 profiles/current.json")
    official.add_argument("--checkpoint", type=Path, help="可选的同一份推理权重文件")
    official.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    args = parser.parse_args()
    try:
        if args.command == "prepare-official":
            prepare(args.archive, args.demo_dir)
            return 0
        if args.command == "visualize":
            export_viewer(args.data, args.run, args.scene, args.out, args.baseline, args.previous, args.replace)
            return 0
        if args.command == "scorecard":
            write_scorecard(args.run, scorecard_for_run(args.run, args.baseline))
            print(f"已写入 {args.run / 'scorecard.csv'} 和 {args.run / 'scorecard.json'}")
            return 0
        config = load_config(args.config)
        handler = partial(process_circle, config=config)
        replay_config = asdict(config)
        if args.tracker == "point-scorer":
            if args.config or getattr(args,"upstream",None) or getattr(args,"original",False):
                parser.error("point-scorer 使用 --profile，不接受基线 --config、--upstream 或 --original")
            from .point_tracker import load_tracker
            handler,replay_config=load_tracker(args.profile,args.checkpoint,args.device)
        elif args.tracker == "track-mt3":
            if not args.checkpoint or not args.upstream or args.config or args.profile:
                parser.error("track-mt3 需要 --checkpoint 和 --upstream，且不接受 --config 或 --profile")
            from .neural import load_tracker
            handler,replay_config=load_tracker(args.checkpoint,args.upstream,args.device)
        elif args.checkpoint or args.profile or getattr(args,"upstream",None):
            parser.error("模型配置和权重参数需要同时指定对应的 --tracker")
        if args.command == "replay":
            scenes = list(dict.fromkeys(args.scene)) if args.scene else None
            if args.split:
                scenes = load_split(args.split)
            if (args.data.is_dir() or args.data.suffix.lower() == ".zip") and not (scenes or args.all):
                parser.error("训练包请指定 --scene、--split 或 --all")
            replay(args.data, args.out, handler, replay_config, scenes)
        else:
            run_official(args.demo_dir, args.input_dir, args.out, args.port, handler, replay_config,
                         original=args.original)
        return 0
    except (OSError, RuntimeError, ValueError, TypeError, EOFError, subprocess.TimeoutExpired) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("运行已中止", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
