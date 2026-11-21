"""Reproducible Houyi-Dense training script (Stage-1 & Stage-2) for MasterTracker V14.

Strictly excludes the 30 `dev` scenes in `splits/v1.json` by default (170 non-dev scenes)
for leak-free local validation. Pass `--all-scenes` only for final 200-scene leaderboard training.

Usage examples (run from `b02-starter/`):
  # Stage 1 (65k sectors, circle % 22 == 14, warm-start from HouyiBypassV2/Model/model.pth or houyi-cv):
  python experiments/master_v14/train_houyi_dense.py \
      --stage 1 --data "path/to/初赛训练数据集.zip" --houyi-dir "path/to/HouyiBypassV2" \
      --init-ckpt "path/to/HouyiBypassV2/Model/model.pth" --out-dir runs/houyi_dense_170

  # Stage 2 (128k disjoint sectors, circle % 11 == 5, warm-start from Stage-1 epoch 2):
  python experiments/master_v14/train_houyi_dense.py \
      --stage 2 --data "path/to/初赛训练数据集.zip" --houyi-dir "path/to/HouyiBypassV2" \
      --init-ckpt runs/houyi_dense_170/houyi-dense-e02.pt --out-dir runs/houyi_dense_170
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STARTER_ROOT = Path(__file__).resolve().parents[2]
if str(STARTER_ROOT) not in sys.path:
    sys.path.insert(0, str(STARTER_ROOT))

from b02.master_tracker import HouyiScorer, SCAN_COUNT, SECTOR_BATCH
from b02.replay import _scans, open_scenes


def collect_sectors(scorer: HouyiScorer, data_path: Path, stride: int, offset: int,
                    min_circle: int, all_scenes: bool = False):
    dev = set() if all_scenes else set(
        json.loads((STARTER_ROOT / "splits" / "v1.json").read_text(encoding="utf-8"))["dev"]
    )
    samples = []
    t0 = time.time()
    with open_scenes(data_path) as (scenes, open_scene):
        pool = [s for s in scenes if s not in dev]
        print(f"[collect] scenes={len(pool)} (excluded_dev={len(dev)}) stride={stride} offset={offset}", flush=True)
        for idx, sc in enumerate(pool, 1):
            history = deque(maxlen=SCAN_COUNT - 1)
            seen_cnt = {}
            with open_scene(sc) as src:
                for circle, points, labels in _scans(src):
                    ranks = np.zeros(len(points), dtype=np.int8)
                    for pi, p in enumerate(points):
                        lb = labels[p["PntID"]]
                        if lb > 0:
                            c = seen_cnt.get(lb, 0) + 1
                            seen_cnt[lb] = c
                            ranks[pi] = 2 if c <= 3 else 1  # 2: birth obs1..3, 1: normal truth
                        elif lb == 0:
                            ranks[pi] = 0  # clutter
                        else:
                            ranks[pi] = -1  # unknown (ignored)
                    if circle >= min_circle and circle % stride == offset and points:
                        for features, rows, pt_idx in scorer._sectors(points, history):
                            chosen_ranks = ranks[pt_idx]
                            if np.any(chosen_ranks >= 0):
                                samples.append((features.astype(np.float16), rows.astype(np.int16), chosen_ranks))
                    history.append(points)
            if idx % 34 == 0 or idx == len(pool):
                print(f"[collect] {idx}/{len(pool)} sectors={len(samples)} ({time.time() - t0:.1f}s)", flush=True)
    return samples


def batch_loss(scorer: HouyiScorer, batch, w_map, label_smoothing: float = 0.01):
    subset = [(f.astype(np.float32), r.astype(np.int64), None) for f, r, _ in batch]
    logits = scorer.forward_sector_batch(subset, model_idx=0)  # (B, L, 2)
    preds, tgts, sws = [], [], []
    for i, (_, rows, ranks) in enumerate(batch):
        keep = ranks >= 0
        if not np.any(keep):
            continue
        rk = ranks[keep]
        preds.append(logits[i, rows[keep]])
        tgts.append(torch.as_tensor(rk > 0, dtype=torch.long, device=scorer.device))
        sws.append(w_map[torch.as_tensor(rk, dtype=torch.long, device=scorer.device)])
    pred_cat = torch.cat(preds, dim=0)
    tgt_cat = torch.cat(tgts, dim=0)
    sw_cat = torch.cat(sws, dim=0)
    per_pt = F.cross_entropy(pred_cat, tgt_cat, reduction="none", label_smoothing=label_smoothing)
    return (per_pt * sw_cat).sum() / sw_cat.sum().clamp_min(1.0)


def step_loss(scorer: HouyiScorer, batch, w_map, label_smoothing: float = 0.01):
    try:
        loss = batch_loss(scorer, batch, w_map, label_smoothing=label_smoothing)
        loss.backward()
        return loss.item(), False
    except torch.OutOfMemoryError:
        scorer.models[0].zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        tot = 0.0
        for s in batch:
            loss = batch_loss(scorer, [s], w_map, label_smoothing=label_smoothing) / len(batch)
            loss.backward()
            tot += loss.item()
        return tot, True


def main():
    parser = argparse.ArgumentParser(description="Train Houyi-Dense Stage-1 or Stage-2 point classifier.")
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True,
                        help="1: stride=22 offset=14 (~65k sectors, lr=4e-5); 2: stride=11 offset=5 (~128k sectors, lr=2e-5)")
    parser.add_argument("--data", type=Path, required=True, help="Training ZIP or CSV directory")
    parser.add_argument("--houyi-dir", type=Path, required=True, help="Path to HouyiBypassV2 directory")
    parser.add_argument("--init-ckpt", type=Path, default=None,
                        help="Warm-start checkpoint (.pt or HouyiBypassV2/Model/model.pth)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--all-scenes", action="store_true",
                        help="Use all 200 scenes (ONLY for final leaderboard weights, NOT for local dev_30 eval)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    init_ckpt = args.init_ckpt or (args.houyi_dir / "Model" / "model.pth")
    if args.stage == 1:
        stride, offset, min_circle = 22, 14, 14
        lr = args.lr or 4e-5
        warmup = 300
        prefix = "houyi-dense"
    else:
        stride, offset, min_circle = 11, 5, 10
        lr = args.lr or 2e-5
        warmup = 400
        prefix = "houyi-dense-v3"
    if args.all_scenes:
        prefix += "-all200"

    torch.manual_seed(args.seed + args.stage * 100)
    scorer = HouyiScorer(args.houyi_dir, init_ckpt, device=args.device)
    model = scorer.models[0]
    print(f"[init] Loaded {init_ckpt} on {scorer.device}", flush=True)

    samples = collect_sectors(scorer, args.data, stride=stride, offset=offset,
                              min_circle=min_circle, all_scenes=args.all_scenes)
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    per_ep = -(-len(samples) // SECTOR_BATCH)
    total_steps = args.epochs * per_ep

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    w_map = torch.tensor([1.0, 2.0, 3.0], device=scorer.device)
    rng = np.random.default_rng(args.seed + 303 * args.stage)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(samples))
        losses_ep, ooms = [], 0
        for step_i, st in enumerate(range(0, len(order), SECTOR_BATCH), 1):
            optimizer.zero_grad(set_to_none=True)
            batch = [samples[idx] for idx in order[st:st + SECTOR_BATCH]]
            lval, fb = step_loss(scorer, batch, w_map, label_smoothing=0.01)
            ooms += int(fb)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            schedule.step()
            losses_ep.append(lval)
            if step_i % 5000 == 0 or step_i == per_ep:
                print(f"  [stage{args.stage}] ep {ep} step {step_i}/{per_ep} "
                      f"loss={np.mean(losses_ep[-500:]):.4f} lr={schedule.get_last_lr()[0]:.2e} "
                      f"oom={ooms} ({time.time() - t0:.0f}s)", flush=True)

        save_path = args.out_dir / f"{prefix}-e{ep:02d}.pt"
        torch.save({
            "format": f"b02-{prefix}",
            "epoch": ep,
            "sectors": len(samples),
            "all_scenes": args.all_scenes,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }, save_path.with_suffix(".tmp"))
        save_path.with_suffix(".tmp").replace(save_path)
        print(f"Saved {save_path} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
