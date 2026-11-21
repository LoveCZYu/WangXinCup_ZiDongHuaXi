"""Collect Closed-Loop DAgger V18 Conflict Components (40-dim edges + 24-scan TrackSeqMemory + Bipartite rel_mat)
across sampled training scenes using Houyi-V18-Joint-12scan.

Strictly uses non-dev training scenes by default; pass `--all-scenes` for 200-scene leaderboard training.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from collections import Counter, deque
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

STARTER_ROOT = Path(__file__).resolve().parents[2]
if str(STARTER_ROOT) not in sys.path:
    sys.path.insert(0, str(STARTER_ROOT))

from b02.master_tracker import (
    MasterTracker, HouyiScorer, FastNumpyConflictNet, TrackSeqMemory,
    _build_v18_conflict_graph, _build_rel_mat, MAX_COMP_EDGES,
    resolve_conflict_checkpoint, resolve_default_paths,
)
from b02.replay import _scans, open_scenes

SCAN_COUNT_V18 = 12


@torch.inference_mode()
def run_v18_single_pass(scorer: HouyiScorer, points, history, cand_thresh_ld: float = -3.3):
    """Single causal pass reusing HouyiScorer's 12s pair+motion pipeline without azimuth TTA."""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.float32), {}, np.zeros((0, 3), dtype=np.float32)
    old_tta = scorer.use_az_tta
    scorer.use_az_tta = False
    try:
        all_keys = None
        if history:
            all_keys = {(int(p["CircleID"]), int(p["PntID"])) for h in history for p in h}
        probs, aff, mot = scorer.score_and_affinity(
            points, history, active_hist_keys=all_keys, cand_thresh=0.025, return_motion=True
        )
        probs_a = np.clip(np.asarray(probs, dtype=np.float32), 1e-6, 1.0 - 1e-6)
        ld_out = (np.log(probs_a / (1.0 - probs_a)) * scorer.temperature).astype(np.float32)
        return ld_out, aff, (mot if mot is not None else np.zeros((n, 3), dtype=np.float32))
    finally:
        scorer.use_az_tta = old_tta


def collect_scene_components(scene_idx: int, frames, scorer: HouyiScorer, v17_cnet):
    """Run closed-loop tracker powered by Houyi-V18-Joint and record labeled conflict components."""
    trk = MasterTracker(conflict_net=v17_cnet)
    memories = {}
    bid_gt_votes = {}
    history = deque(maxlen=SCAN_COUNT_V18 - 1)

    comp_list = []
    n_edges_tot = 0
    n_neg_tot = 0

    for circle, points, labels in frames:
        if not points:
            history.append(points)
            continue
        ld, aff, mot = run_v18_single_pass(scorer, points, history)
        probs = (1.0 / (1.0 + np.exp(-np.clip(ld / 0.90, -30.0, 30.0)))).tolist()

        cand_pts, cand_scores, cand_orig = [], [], []
        for orig_i, (p, s) in enumerate(zip(points, probs)):
            if s >= trk.min_cand_score:
                cand_pts.append(p)
                cand_scores.append(s)
                cand_orig.append(orig_i)

        trk._cur_aff = aff
        trk._cand_orig_idx = cand_orig
        confirmed = [b for b, t in trk.tracks.items() if trk._is_confirmed(t)]
        next_scan = trk.scan + 1
        if confirmed and cand_pts:
            edges, track_seqs, components = _build_v18_conflict_graph(
                trk, memories, confirmed, cand_pts, cand_scores,
                scan=next_scan, gate_scale=1.0, occupied=set(), cur_motion=mot,
            )
            bid_true_gt = {}
            for b in confirmed:
                vc = bid_gt_votes.get(b)
                if vc:
                    top_gt, top_cnt = vc.most_common(1)[0]
                    tot_cnt = sum(vc.values())
                    if top_gt > 0 and top_cnt >= 2 and top_cnt / tot_cnt >= 0.55:
                        bid_true_gt[b] = top_gt

            for comp in components:
                if len(comp) == 1 and (circle % 5 != 0):
                    continue
                if len(comp) > MAX_COMP_EDGES:
                    comp = sorted(comp, key=lambda ei: edges[ei][2][5])[:MAX_COMP_EDGES]

                c_feats, c_seqs, c_bids, c_cis, c_y = [], [], [], [], []
                has_known = False
                for ei in comp:
                    bid, ci, feat, _ = edges[ei]
                    pt_lbl = int(labels[cand_pts[ci]["PntID"]])
                    trk_gt = bid_true_gt.get(bid)
                    if trk_gt is None or pt_lbl < 0:
                        y = -1.0
                    else:
                        y = 1.0 if (pt_lbl == trk_gt) else 0.0
                        has_known = True
                    c_feats.append(feat)
                    c_seqs.append(track_seqs[bid])
                    c_bids.append(bid)
                    c_cis.append(ci)
                    c_y.append(y)
                if not has_known:
                    continue
                rel = _build_rel_mat(c_bids, c_cis)
                y_arr = np.asarray(c_y, dtype=np.float32)
                comp_list.append((
                    np.stack(c_feats, axis=0).astype(np.float16),
                    np.stack(c_seqs, axis=0).astype(np.float16),
                    rel,
                    y_arr,
                    int(scene_idx),
                    int(len(comp) >= 2),
                ))
                n_edges_tot += int(np.sum(y_arr >= 0))
                n_neg_tot += int(np.sum(y_arr == 0.0))

        prev_tracks = set(trk.tracks.keys())
        outs = trk(points, probs, pair_aff=aff)
        out_by_bid = {o["BatchID"]: o for o in outs}
        pt_by_pid = {p["PntID"]: (p, s) for p, s in zip(points, probs)}

        for bid, t in trk.tracks.items():
            if bid not in prev_tracks:
                p_init = {"Azimuth": t.az, "Elevation": t.raw_el, "Distance": t.r, "Amp": t.amp, "SNR": t.snr}
                memories[bid] = TrackSeqMemory(p_init, t.last_score, trk.scan)
            elif t.real_last_scan == trk.scan:
                o = out_by_bid.get(bid)
                if o is not None and o["PntID"] in pt_by_pid:
                    p_hit, sc_hit = pt_by_pid[o["PntID"]]
                else:
                    p_hit = {"Azimuth": t.az, "Elevation": t.raw_el, "Distance": t.r, "Amp": t.amp, "SNR": t.snr}
                    sc_hit = t.last_score
                memories[bid].record_hit(t, p_hit, sc_hit, trk.scan)
            else:
                memories[bid].record_miss(t, trk.scan)

        for o in outs:
            lb = int(labels[o["PntID"]])
            if lb > 0:
                bid_gt_votes.setdefault(o["BatchID"], Counter())[lb] += 1

        if trk.scan % 25 == 0:
            alive = set(trk.tracks.keys()) | set(trk.dormant.keys())
            memories = {b: m for b, m in memories.items() if b in alive}

        history.append(points)

    return comp_list, n_edges_tot, n_neg_tot


def main():
    ap = argparse.ArgumentParser(description="Collect closed-loop DAgger conflict components for V18-C SeqCrossAttnNet.")
    ap.add_argument("--data-zip", type=Path, required=True, help="Path to training ZIP or CSV directory")
    ap.add_argument("--houyi-dir", type=Path, default=None, help="Path to HouyiBypassV2 directory")
    ap.add_argument("--v18-ckpt", type=Path, required=True, help="Trained houyi-v18-joint-12scan.pt checkpoint")
    ap.add_argument("--conflict-ckpt", type=Path, default=None, help="Optional V17 conflict MLP to guide rollout")
    ap.add_argument("--out", type=Path, required=True, help="Output .pkl path")
    ap.add_argument("--num-scenes", type=int, default=30, help="Uniformly sampled scenes for DAgger rollout")
    ap.add_argument("--all-scenes", action="store_true", help="Sample across all 200 scenes instead of 170 non-dev scenes")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t0 = time.time()
    houyi_dir = args.houyi_dir or resolve_default_paths()[0]
    dev = set() if args.all_scenes else set(
        json.loads((STARTER_ROOT / "splits" / "v1.json").read_text(encoding="utf-8"))["dev"]
    )
    scorer = HouyiScorer(
        houyi_dir, args.v18_ckpt, device=args.device, temperature=0.90,
        pair_checkpoint=args.v18_ckpt, use_az_tta=False,
    )
    c_path = args.conflict_ckpt if args.conflict_ckpt else resolve_conflict_checkpoint()
    v17_cnet = FastNumpyConflictNet(c_path) if (c_path and Path(c_path).is_file()) else None

    all_comps = []
    tot_e = tot_n = 0
    with open_scenes(args.data_zip) as (scenes, open_scene):
        pool = [s for s in scenes if s not in dev]
        n_pick = min(args.num_scenes, len(pool))
        sel_idx = np.linspace(0, len(pool) - 1, n_pick, dtype=int)
        chosen = [pool[i] for i in sel_idx]
        print(f"[V18-DAgger] Collecting conflict components from {len(chosen)} scenes (dev excluded={len(dev)})...", flush=True)
        for s_i, sc in enumerate(chosen):
            with open_scene(sc) as src:
                frames = list(_scans(src))
            comps, ne, nn_ = collect_scene_components(s_i, frames, scorer, v17_cnet)
            all_comps.extend(comps)
            tot_e += ne
            tot_n += nn_
            print(f"  [V18-DAgger] {s_i+1:2d}/{len(chosen)} {sc}: comps={len(comps):5d} edges={ne:6d} "
                  f"(neg={nn_:5d}) | total_comps={len(all_comps):6d} ({(time.time()-t0)/60:.1f}m)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump({"comps": all_comps, "scenes": chosen}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[DONE] Saved {args.out} ({args.out.stat().st_size/1e6:.1f} MB): "
          f"comps={len(all_comps)}, labeled_edges={tot_e}, neg_rate={tot_n/max(1,tot_e):.4f}", flush=True)


if __name__ == "__main__":
    main()
