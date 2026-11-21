"""Train Houyi-V18-Joint-12scan: Full-Coverage (100% circles via rotating stride=3 x 3 epochs)
Multi-Task 51M Houyi Transformer with:
  - Head 1: Hard-Birth + Dense-Clutter Focal Point Classifier (12-scan causal context)
  - Head 2: BackwardMotionHead (CenterTrack-style per-scan backward velocity regression d_az, d_el, d_r)
  - Head 3: HouyiPairHead (12-scan Tight-Formation Hard-Negative Pair Affinity + SupCon Metric Embedding)
  - Dense-clutter/noise Domain Randomization: 360-deg Az shift (75%), Heavy Clutter Injection (45%),
    Historical Wingman Dropout (20%), Mild Sensor/Amp/SNR Jitter (35%).

Strictly uses the 170 non-dev scenes from `splits/v1.json` by default (zero leakage to dev_30).
Pass `--all-scenes` ONLY for final 200-scene leaderboard weights after local verification.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from collections import deque
import json
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

from b02.master_tracker import (
    HouyiScorer, HouyiPairHead, BackwardMotionHead, rel_geom_12s_np, wrap_az, resolve_default_paths,
)
from b02.replay import _scans, open_scenes

SCAN_COUNT_V18 = 12
SEED = 20261120


def collect_v18_epoch_sectors(scorer: HouyiScorer, data_zip: Path, stride: int, offset: int,
                              all_scenes: bool = False):
    """Collect 12-scan sectors with full labels (cls + backward motion + 12-scan pair affinity)."""
    dev = set() if all_scenes else set(
        json.loads((STARTER_ROOT / "splits" / "v1.json").read_text(encoding="utf-8"))["dev"]
    )
    samples = []
    t0 = time.time()
    with open_scenes(data_zip) as (scenes, open_scene):
        pool = [s for s in scenes if s not in dev]
        print(f"[V18-collect] scenes={len(pool)} stride={stride} offset={offset}", flush=True)
        for idx, sc in enumerate(pool, 1):
            history_pts = deque(maxlen=SCAN_COUNT_V18 - 1)
            history_lbls = deque(maxlen=SCAN_COUNT_V18 - 1)
            seen_cnt = {}
            with open_scene(sc) as src:
                for circle, points, labels in _scans(src):
                    cur_lbl_arr = np.empty(len(points), dtype=np.int32)
                    cur_ranks = np.zeros(len(points), dtype=np.int8)
                    cur_low_snr = np.zeros(len(points), dtype=np.bool_)
                    for pi, p in enumerate(points):
                        lb = int(labels[p["PntID"]])
                        cur_lbl_arr[pi] = lb
                        if lb > 0:
                            c = seen_cnt.get(lb, 0) + 1
                            seen_cnt[lb] = c
                            cur_ranks[pi] = 3 if c <= 2 else (2 if c == 3 else 1)
                            if p["SNR"] < 0.35:
                                cur_low_snr[pi] = True
                        elif lb == 0:
                            cur_ranks[pi] = 0
                        else:
                            cur_ranks[pi] = -1

                    if circle >= 13 and (circle % stride == offset) and points and len(history_pts) >= 4:
                        sectors, _, _, first_current = scorer._sectors(points, history_pts, return_meta=True)
                        cloud_lbl = np.concatenate(list(history_lbls) + [cur_lbl_arr])
                        for feats, cur_rows, pt_idx, members in sectors:
                            if len(feats) > 1550:
                                continue
                            chosen_ranks = cur_ranks[pt_idx]
                            if not np.any(chosen_ranks >= 0):
                                continue
                            m_lbl = cloud_lbl[members]
                            is_cur_mask = (members >= first_current)
                            samples.append((
                                feats.astype(np.float16),
                                cur_rows.astype(np.int16),
                                chosen_ranks,
                                cur_low_snr[pt_idx],
                                m_lbl.astype(np.int32),
                                is_cur_mask,
                            ))
                    history_pts.append(points)
                    history_lbls.append(cur_lbl_arr)
            if idx % 34 == 0 or idx == len(pool):
                print(f"  [V18-collect] {idx}/{len(pool)} sectors={len(samples)} ({time.time()-t0:.1f}s)", flush=True)
    return samples


def augment_v18_sector(feats_f16, cur_rows_i16, ranks, low_snr, m_lbl, is_cur_mask, rng: np.random.Generator):
    """Physics-consistent domain randomization preserving exact labels for all 3 heads."""
    f = feats_f16.astype(np.float32)
    cur_rows = cur_rows_i16.astype(np.int64)
    lbl = m_lbl
    cur_m = is_cur_mask

    if rng.random() < 0.75:
        f = f.copy()
        f[:, 4] = (f[:, 4] + float(rng.random())) % 1.0

    if rng.random() < 0.35:
        n = len(f)
        f[:, 1] = np.clip(f[:, 1] + rng.normal(0.0, 0.0022, n), -0.1, 1.1)
        f[:, 2] = np.clip(f[:, 2] + rng.normal(0.0, 0.0022, n), -0.1, 1.1)
        f[:, 3] = np.clip(f[:, 3] + rng.normal(0.0, 0.00016, n), 0.001, 1.0)
        f[:, 4] = (f[:, 4] + rng.normal(0.0, 0.00040, n)) % 1.0
        f[:, 5] = np.clip(f[:, 5] + rng.normal(0.0, 0.0020, n), 0.0, 1.0)
        amp_s = float(rng.uniform(0.94, 1.06))
        f[:, 7] = np.clip(f[:, 7] * amp_s + rng.normal(0.0, 0.014, n), 0.0, 1.0)
        f[:, 8] = np.clip(f[:, 8] * amp_s + rng.normal(0.0, 0.014, n), 0.0, 1.0)

    if rng.random() < 0.20 and len(f) > 35:
        drop_rate = float(rng.uniform(0.10, 0.22))
        keep = cur_m | (rng.random(len(f)) > drop_rate)
        if keep.sum() >= len(cur_rows) + 8:
            remap = np.cumsum(keep) - 1
            f = f[keep]
            cur_rows = remap[cur_rows]
            lbl = lbl[keep]
            cur_m = cur_m[keep]

    if rng.random() < 0.45 and len(f) < 1350:
        frac = float(rng.uniform(0.12, 0.38))
        n_inj = min(140, max(6, int(len(f) * frac)))
        t_max = max(float(f[:, 0].max()), 0.1)
        syn = np.empty((n_inj, 10), dtype=np.float32)
        syn[:, 0] = rng.uniform(0.0, t_max, n_inj)
        syn[:, 1] = rng.uniform(0.02, 0.98, n_inj)
        syn[:, 2] = rng.uniform(0.02, 0.98, n_inj)
        syn[:, 3] = rng.uniform(0.08, 0.95, n_inj)
        az_lo, az_hi = float(f[:, 4].min()), float(f[:, 4].max())
        syn[:, 4] = rng.uniform(az_lo, az_hi, n_inj) if (az_hi - az_lo < 0.5) else rng.uniform(0.0, 1.0, n_inj)
        syn[:, 5] = rng.exponential(0.035, n_inj).clip(0.0, 0.35)
        syn[:, 6] = 0.0
        syn[:, 7] = rng.uniform(0.12, 0.55, n_inj)
        syn[:, 8] = syn[:, 7]
        syn[:, 9] = 0.5
        f = np.concatenate((f, syn), axis=0)
        lbl = np.concatenate((lbl, np.zeros(n_inj, dtype=np.int32)), axis=0)
        cur_m = np.concatenate((cur_m, np.zeros(n_inj, dtype=bool)), axis=0)

    return f, cur_rows, ranks, low_snr, lbl, cur_m


def mine_pairs_and_motion(feats: np.ndarray, cur_rows: np.ndarray, m_lbl: np.ndarray, is_cur_mask: np.ndarray):
    """Extract 12-scan hard-negative pairs for PairHead and backward velocity targets for MotionHead."""
    hist_idx = np.flatnonzero(~is_cur_mask)
    empty_p = (np.empty((0, 2), dtype=np.int16), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32))
    empty_m = (np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32))
    if not len(hist_idx) or not len(cur_rows):
        return empty_p, empty_m

    hist_tgt = hist_idx[m_lbl[hist_idx] > 0]
    hist_clut = hist_idx[m_lbl[hist_idx] == 0]
    if not len(hist_tgt):
        return empty_p, empty_m

    h_az = feats[hist_tgt, 4]
    h_r = feats[hist_tgt, 3]
    h_lbl = m_lbl[hist_tgt]
    hc_az = feats[hist_clut, 4] if len(hist_clut) else None
    hc_r = feats[hist_clut, 3] if len(hist_clut) else None

    pairs, p_tags, p_wts = [], [], []
    m_rows, m_tgts, m_wts = [], [], []

    for cr in cur_rows:
        lb = int(m_lbl[cr])
        if lb < 0:
            continue
        c_az = float(feats[cr, 4])
        c_r = float(feats[cr, 3])
        c_el = float(feats[cr, 5])
        c_t = float(feats[cr, 0])

        dr = np.abs(h_r - c_r)
        daz = np.abs(h_az - c_az)
        daz = np.minimum(daz, 1.0 - daz)
        near_m = (dr < 0.036) & (daz < 0.042)
        if not np.any(near_m):
            continue
        near_pos = np.flatnonzero(near_m)
        near_rows = hist_tgt[near_pos]
        near_lbls = h_lbl[near_pos]

        if lb > 0:
            same = near_rows[near_lbls == lb]
            diff = near_rows[near_lbls != lb]
            is_tight_form = False
            if len(diff):
                d_dr = np.abs(feats[diff, 3] - c_r)
                d_daz = np.abs(feats[diff, 4] - c_az)
                d_daz = np.minimum(d_daz, 1.0 - d_daz)
                is_tight_form = bool(np.any((d_dr < 0.015) & (d_daz < 0.018)))

            if len(same):
                order = np.argsort(feats[same, 0])[::-1]
                same_sorted = same[order]
                for hr in same_sorted[:6]:
                    pairs.append((cr, hr))
                    p_tags.append(1.0)
                    p_wts.append(1.4 if is_tight_form else 1.0)

                hr_prev = int(same_sorted[0])
                dt_scan = (c_t - float(feats[hr_prev, 0])) * 7.0
                if 0.5 <= dt_scan <= 3.5:
                    vaz = wrap_az(c_az - float(feats[hr_prev, 4])) / dt_scan
                    vel = (c_el - float(feats[hr_prev, 5])) / dt_scan
                    vr = (c_r - float(feats[hr_prev, 3])) / dt_scan
                    if abs(vaz) < 0.08 and abs(vr) < 0.06:
                        m_rows.append(int(cr))
                        m_tgts.append((
                            max(-5.0, min(5.0, vaz * 50.0)),
                            max(-5.0, min(5.0, vel * 15.0)),
                            max(-5.0, min(5.0, vr * 80.0)),
                        ))
                        m_wts.append(2.2 if is_tight_form else 1.0)

            if len(diff):
                d_dr = np.abs(feats[diff, 3] - c_r)
                d_daz = np.abs(feats[diff, 4] - c_az)
                d_daz = np.minimum(d_daz, 1.0 - d_daz)
                for idx_d in np.argsort(d_dr + d_daz)[:8]:
                    hr = int(diff[idx_d])
                    is_tf = (d_dr[idx_d] < 0.015) and (d_daz[idx_d] < 0.018)
                    pairs.append((cr, hr))
                    p_tags.append(0.0)
                    p_wts.append(3.0 if is_tf else 1.3)

            if hc_az is not None and len(hist_clut):
                cdr = np.abs(hc_r - c_r)
                cdaz = np.abs(hc_az - c_az)
                cdaz = np.minimum(cdaz, 1.0 - cdaz)
                for idx_c in np.flatnonzero((cdr < 0.020) & (cdaz < 0.024))[:2]:
                    pairs.append((cr, int(hist_clut[idx_c])))
                    p_tags.append(0.0)
                    p_wts.append(1.0)
        else:
            for hr in near_rows[:2]:
                pairs.append((cr, int(hr)))
                p_tags.append(0.0)
                p_wts.append(1.4)

    p_out = (
        (np.asarray(pairs, dtype=np.int16), np.asarray(p_tags, dtype=np.float32), np.asarray(p_wts, dtype=np.float32))
        if pairs else empty_p
    )
    m_out = (
        (np.asarray(m_rows, dtype=np.int64), np.asarray(m_tgts, dtype=np.float32), np.asarray(m_wts, dtype=np.float32))
        if m_rows else empty_m
    )
    return p_out, m_out


def build_length_buckets(samples, rng: np.random.Generator):
    n = len(samples)
    keys = np.asarray([len(s[0]) for s in samples], dtype=np.float32) + rng.uniform(-35.0, 35.0, n)
    order = np.argsort(keys)
    batches = []
    i = 0
    while i < n:
        L = len(samples[order[i]][0])
        bs = 4 if L <= 560 else (2 if L <= 1050 else 1)
        batches.append(order[i:i + bs].tolist())
        i += bs
    rng.shuffle(batches)
    return batches


def forward_v18_batch(scorer: HouyiScorer, pair_head: HouyiPairHead, motion_head: BackwardMotionHead,
                      batch_items, rng: np.random.Generator, do_aug: bool = True):
    aug_items = []
    for item in batch_items:
        if do_aug:
            f, cr, rk, ls, ml, cm = augment_v18_sector(*item, rng)
        else:
            f, cr, rk, ls, ml, cm = item[0].astype(np.float32), item[1].astype(np.int64), item[2], item[3], item[4], item[5]
        p_out, m_out = mine_pairs_and_motion(f, cr, ml, cm)
        aug_items.append((f, cr, rk, ls, p_out, m_out))

    B = len(aug_items)
    max_len = max(len(x[0]) for x in aug_items)
    input_data = np.zeros((B, max_len, 10), dtype=np.float32)
    valid = np.zeros((B, max_len), dtype=bool)
    for b, (f, _, _, _, _, _) in enumerate(aug_items):
        input_data[b, :len(f)] = f
        valid[b, :len(f)] = True

    raw = torch.from_numpy(input_data).to(scorer.device)
    mask = torch.from_numpy(valid).to(scorer.device)
    inputs = F.pad(torch.cat((raw, scorer.augmenter.augment(raw)), dim=-1), (0, scorer.extra))
    model = scorer.models[0]

    with torch.autocast(scorer.device.type, dtype=torch.bfloat16, enabled=scorer.device.type == "cuda"):
        hidden = model.input_embedding(inputs)
        attention = (~mask[:, None, None]).to(hidden.dtype) * -10000.0
        encoded = model.bert_encoder(hidden, attention_mask=attention, return_dict=True).last_hidden_state
        direct = model.point_classifier_input_emb(inputs)
        combined = model.point_classifier_concat_emb(torch.cat((encoded, direct), dim=-1))
        cls_logits = model.point_classifier(combined).float()

    cls_losses, cls_weights = [], []
    for b, (_, cr, rk, ls, _, _) in enumerate(aug_items):
        keep = rk >= 0
        if not np.any(keep):
            continue
        r_idx = torch.from_numpy(cr[keep]).to(scorer.device)
        r_rk = rk[keep]
        r_ls = ls[keep]
        tgt = torch.from_numpy((r_rk > 0).astype(np.int64)).to(scorer.device)
        w = np.where(r_rk == 0, 1.0, np.where(r_rk == 1, 2.0, np.where(r_rk == 2, 3.0, 4.2))).astype(np.float32)
        w = np.where(r_ls & (r_rk > 0), w * 1.35, w)
        w_t = torch.from_numpy(w).to(scorer.device)
        lg = cls_logits[b, r_idx]
        ce = F.cross_entropy(lg, tgt, reduction="none", label_smoothing=0.01)
        pt = torch.softmax(lg.detach(), dim=-1).gather(1, tgt.unsqueeze(1)).squeeze(1)
        focal = (1.0 - pt).clamp(min=1e-4).pow(0.5)
        cls_losses.append(ce * focal * w_t)
        cls_weights.append(w_t)
    loss_cls = (torch.cat(cls_losses).sum() / torch.cat(cls_weights).sum().clamp(min=1.0)) if cls_losses else None

    mot_preds = motion_head(encoded, direct)
    m_losses, m_wts_t = [], []
    for b, (_, _, _, _, _, (mr, mt, mw)) in enumerate(aug_items):
        if not len(mr):
            continue
        r_t = torch.from_numpy(mr).to(scorer.device)
        t_t = torch.from_numpy(mt).to(scorer.device)
        w_t = torch.from_numpy(mw).to(scorer.device)
        sl1 = F.smooth_l1_loss(mot_preds[b, r_t], t_t, beta=0.15, reduction="none").mean(dim=-1)
        m_losses.append(sl1 * w_t)
        m_wts_t.append(w_t)
    loss_mot = (torch.cat(m_losses).sum() / torch.cat(m_wts_t).sum().clamp(min=1.0)) if m_losses else None

    z_all = pair_head.embed(encoded, direct)
    p_losses, c_losses, p_wts_t = [], [], []
    for b, (f, _, _, _, (prs, ptg, pwt), _) in enumerate(aug_items):
        if not len(prs):
            continue
        c_i = prs[:, 0].astype(np.int64)
        h_i = prs[:, 1].astype(np.int64)
        gm = torch.from_numpy(rel_geom_12s_np(f[c_i], f[h_i])).to(scorer.device)
        c_t = torch.from_numpy(c_i).to(scorer.device)
        h_t = torch.from_numpy(h_i).to(scorer.device)
        y_t = torch.from_numpy(ptg).to(scorer.device)
        w_t = torch.from_numpy(pwt).to(scorer.device)
        pl, cl = pair_head.score_pairs(z_all[b, c_t], z_all[b, h_t], gm)
        p_losses.append(F.binary_cross_entropy_with_logits(pl, y_t, reduction="none") * w_t)
        c_losses.append(F.binary_cross_entropy_with_logits(cl, y_t, reduction="none") * w_t)
        p_wts_t.append(w_t)
    if p_losses:
        wsum = torch.cat(p_wts_t).sum().clamp(min=1.0)
        loss_pair = torch.cat(p_losses).sum() / wsum
        loss_cos = torch.cat(c_losses).sum() / wsum
    else:
        loss_pair = loss_cos = None

    return loss_cls, loss_mot, loss_pair, loss_cos


@torch.inference_mode()
def quick_val_dev30(scorer: HouyiScorer, pair_head: HouyiPairHead, motion_head: BackwardMotionHead, val_scenes: dict):
    model = scorer.models[0]
    model.eval(); pair_head.eval(); motion_head.eval()
    tp = fp = fn = 0
    obs1_tp = obs1_tot = 0
    pair_pos, pair_neg, mot_errs = [], [], []

    for frames in val_scenes.values():
        h_pts = deque(maxlen=SCAN_COUNT_V18 - 1)
        h_lbl = deque(maxlen=SCAN_COUNT_V18 - 1)
        seen = set()
        for ci, (circle, pts, lbls) in enumerate(frames):
            cur_lbl = np.asarray([int(lbls[p["PntID"]]) for p in pts], dtype=np.int32)
            is_obs1 = np.asarray([(lb > 0 and lb not in seen) for lb in cur_lbl], dtype=bool)
            for lb in cur_lbl:
                if lb > 0:
                    seen.add(int(lb))
            if ci >= 15 and ci % 25 == 0 and pts and len(h_pts) >= 6:
                sectors, _, _, first_cur = scorer._sectors(pts, h_pts, return_meta=True)
                cloud_lbl = np.concatenate(list(h_lbl) + [cur_lbl])
                for feats, cur_rows, pt_idx, members in sectors:
                    if len(feats) > 1550:
                        continue
                    raw = torch.from_numpy(feats[None]).to(scorer.device)
                    inputs = F.pad(torch.cat((raw, scorer.augmenter.augment(raw)), dim=-1), (0, scorer.extra))
                    with torch.autocast(scorer.device.type, dtype=torch.bfloat16, enabled=scorer.device.type == "cuda"):
                        h = model.input_embedding(inputs)
                        enc = model.bert_encoder(h, return_dict=True).last_hidden_state
                        d = model.point_classifier_input_emb(inputs)
                        comb = model.point_classifier_concat_emb(torch.cat((enc, d), dim=-1))
                        lg = model.point_classifier(comb).float()[0]
                    ld = (lg[cur_rows, 1] - lg[cur_rows, 0]).cpu().numpy()
                    pred = ld > 0.0
                    gt = cur_lbl[pt_idx]
                    valid = gt >= 0
                    tp += int(np.sum(pred[valid] & (gt[valid] > 0)))
                    fp += int(np.sum(pred[valid] & (gt[valid] == 0)))
                    fn += int(np.sum((~pred[valid]) & (gt[valid] > 0)))
                    o1 = is_obs1[pt_idx]
                    obs1_tp += int(np.sum(pred[o1]))
                    obs1_tot += int(np.sum(o1))

                    m_lbl = cloud_lbl[members]
                    is_cm = (members >= first_cur)
                    (prs, ptg, pwt), (mr, mt, _) = mine_pairs_and_motion(feats, cur_rows, m_lbl, is_cm)
                    if len(prs):
                        z = pair_head.embed(enc, d)[0]
                        ci_i = prs[:, 0].astype(np.int64)
                        hi_i = prs[:, 1].astype(np.int64)
                        gm = torch.from_numpy(rel_geom_12s_np(feats[ci_i], feats[hi_i])).to(scorer.device)
                        pl, _ = pair_head.score_pairs(
                            z[torch.from_numpy(ci_i).to(scorer.device)],
                            z[torch.from_numpy(hi_i).to(scorer.device)], gm,
                        )
                        pp = torch.sigmoid(pl).cpu().numpy()
                        tf_mask = (ptg > 0.5) | (pwt >= 2.5)
                        for yv, pv in zip(ptg[tf_mask], pp[tf_mask]):
                            (pair_pos if yv > 0.5 else pair_neg).append(float(pv))
                    if len(mr):
                        mp = motion_head(enc, d)[0, torch.from_numpy(mr).to(scorer.device)].cpu().numpy()
                        mot_errs.append(float(np.mean(np.abs(mp - mt))))
            h_pts.append(pts)
            h_lbl.append(cur_lbl)

    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    o1_r = obs1_tp / max(1, obs1_tot)
    auc = 0.5
    if pair_pos and pair_neg:
        pos_a = np.asarray(pair_pos)
        neg_a = np.asarray(pair_neg)
        all_s = np.concatenate((pos_a, neg_a))
        ord_i = np.argsort(all_s)
        ranks = np.empty_like(ord_i, dtype=np.float64)
        ranks[ord_i] = np.arange(1, len(all_s) + 1)
        auc = float((ranks[:len(pos_a)].sum() - len(pos_a) * (len(pos_a) + 1) / 2.0) / (len(pos_a) * len(neg_a)))
    mae = float(np.mean(mot_errs)) if mot_errs else 0.0
    model.train(); pair_head.train(); motion_head.train()
    return {"f1": f1, "obs1_rec": o1_r, "tf_auc": auc, "mot_mae": mae}


def main():
    ap = argparse.ArgumentParser(description="Train Houyi-V18-Joint-12scan (3 heads: Cls + BackwardMotion + 12s PairHead).")
    ap.add_argument("--data-zip", type=Path, required=True, help="Path to training ZIP or CSV directory")
    ap.add_argument("--houyi-dir", type=Path, default=None, help="Path to HouyiBypassV2 directory (or set B02_HOUYI_DIR)")
    ap.add_argument("--init-ckpt", type=Path, required=True, help="Warm-start checkpoint (.pt or Model/model.pth)")
    ap.add_argument("--pair-init", type=Path, default=None, help="Optional houyi-pair-e02.pt to warm-start PairHead")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for checkpoints")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--all-scenes", action="store_true", help="Train on all 200 scenes ONLY for final leaderboard submission")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    houyi_dir = args.houyi_dir or resolve_default_paths()[0]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = args.out_dir / ("houyi-v18-joint-12scan-all200.pt" if args.all_scenes else "houyi-v18-joint-12scan.pt")

    print(f"[V18-TRAIN] init={args.init_ckpt} out={out_ckpt} all_scenes={args.all_scenes} "
          f"epochs={args.epochs} stride={args.stride}", flush=True)

    scorer = HouyiScorer(houyi_dir, args.init_ckpt, device=args.device, use_az_tta=False)
    model = scorer.models[0]
    pair_head = HouyiPairHead(in_dim=1024, emb_dim=64, geom_dim=8).to(scorer.device)
    motion_head = BackwardMotionHead(in_dim=1024, hidden=256).to(scorer.device)

    if args.pair_init and args.pair_init.is_file():
        pck = torch.load(args.pair_init, map_location="cpu", weights_only=False)
        psd = pck.get("pair_head_state") or pck.get("pair_head")
        if psd is not None:
            pair_head.load_state_dict(psd)
            print(f"[V18-TRAIN] Warm-started pair_head from {args.pair_init.name}", flush=True)

    val_scenes = {}
    if not args.all_scenes:
        probe_names = ["train_015", "train_021", "train_035", "train_038", "train_065", "train_123"]
        with open_scenes(args.data_zip) as (scenes, open_scene):
            for sc in probe_names:
                if sc in scenes:
                    with open_scene(sc) as src:
                        val_scenes[sc] = list(_scans(src))

    samples_ep0 = collect_v18_epoch_sectors(
        scorer, args.data_zip, stride=args.stride, offset=0, all_scenes=args.all_scenes
    )

    print(f"\n[Phase 1] Warming up MotionHead + PairHead ({args.warmup_steps} steps, backbone frozen)...", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval(); pair_head.train(); motion_head.train()
    opt_w = torch.optim.AdamW(
        list(pair_head.parameters()) + list(motion_head.parameters()), lr=1e-3, weight_decay=1e-4
    )
    w_batches = build_length_buckets(samples_ep0, rng)
    t_w = time.time()
    step = 0
    l_m_acc = l_p_acc = 0.0
    for b_idx in w_batches:
        if step >= args.warmup_steps:
            break
        items = [samples_ep0[i] for i in b_idx]
        _, lm, lp, lc = forward_v18_batch(scorer, pair_head, motion_head, items, rng, do_aug=True)
        tot = None
        if lm is not None:
            tot = 0.5 * lm
            l_m_acc += float(lm.item())
        if lp is not None:
            p_term = 0.7 * lp + 0.3 * lc
            tot = p_term if tot is None else (tot + p_term)
            l_p_acc += float(lp.item())
        if tot is None or not tot.requires_grad:
            continue
        opt_w.zero_grad(set_to_none=True)
        tot.backward()
        nn.utils.clip_grad_norm_(list(pair_head.parameters()) + list(motion_head.parameters()), 1.0)
        opt_w.step()
        step += 1
        if step % 250 == 0:
            print(f"  [Warmup] step {step}/{args.warmup_steps} | mot={l_m_acc/250:.4f} pair={l_p_acc/250:.4f} ({time.time()-t_w:.1f}s)", flush=True)
            l_m_acc = l_p_acc = 0.0

    for p in model.parameters():
        p.requires_grad_(True)
    model.train(); pair_head.train(); motion_head.train()

    est_steps_per_ep = len(w_batches)
    total_steps = est_steps_per_ep * args.epochs
    opt = torch.optim.AdamW([
        {"params": model.parameters(), "lr": 1.8e-5, "weight_decay": 1e-4},
        {"params": list(pair_head.parameters()) + list(motion_head.parameters()), "lr": 3.0e-4, "weight_decay": 1e-4},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=1.5e-6)

    best_score = -1.0
    grad_accum = 2
    t_start = time.time()

    for ep in range(args.epochs):
        if ep == 0:
            samples = samples_ep0
        else:
            del samples
            samples = collect_v18_epoch_sectors(
                scorer, args.data_zip, stride=args.stride, offset=ep % args.stride, all_scenes=args.all_scenes
            )
        batches = build_length_buckets(samples, rng)
        print(f"\n[Epoch {ep+1}/{args.epochs}] sectors={len(samples)} batches={len(batches)} (offset={ep % args.stride})", flush=True)

        c_acc = m_acc = p_acc = 0.0
        n_log = 0
        opt.zero_grad(set_to_none=True)
        t_ep = time.time()

        for bi, b_idx in enumerate(batches, 1):
            items = [samples[i] for i in b_idx]
            lc, lm, lp, lcos = forward_v18_batch(scorer, pair_head, motion_head, items, rng, do_aug=True)
            tot = None
            if lc is not None:
                tot = lc
                c_acc += float(lc.item())
            if lm is not None:
                tot = (0.35 * lm) if tot is None else (tot + 0.35 * lm)
                m_acc += float(lm.item())
            if lp is not None:
                pt = 0.65 * lp + 0.25 * lcos
                tot = pt if tot is None else (tot + pt)
                p_acc += float(lp.item())
            if tot is None or not tot.requires_grad:
                continue
            (tot / grad_accum).backward()
            n_log += 1

            if bi % grad_accum == 0 or bi == len(batches):
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(pair_head.parameters()) + list(motion_head.parameters()), 1.0
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
            sched.step()

            if bi % 2500 == 0 or bi == len(batches):
                denom = max(1, n_log)
                print(f"  [ep {ep+1}] {bi:5d}/{len(batches)} | cls={c_acc/denom:.4f} mot={m_acc/denom:.4f} "
                      f"pair={p_acc/denom:.4f} lr={sched.get_last_lr()[0]:.2e} ({time.time()-t_ep:.1f}s)", flush=True)
                c_acc = m_acc = p_acc = 0.0
                n_log = 0

        val_info = {}
        if val_scenes:
            val_info = quick_val_dev30(scorer, pair_head, motion_head, val_scenes)
            comb_metric = val_info["f1"] + 0.15 * val_info["tf_auc"] + 0.05 * val_info["obs1_rec"] - 0.02 * val_info["mot_mae"]
            print(f"[VAL ep {ep+1}] F1={val_info['f1']:.4f} obs1_rec={val_info['obs1_rec']:.3f} "
                  f"tight_pair_AUC={val_info['tf_auc']:.4f} mot_MAE={val_info['mot_mae']:.4f} "
                  f"(elapsed={(time.time()-t_start)/60:.1f}m)", flush=True)
        else:
            comb_metric = float(ep + 1)

        payload = {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "pair_head_state": {k: v.detach().cpu() for k, v in pair_head.state_dict().items()},
            "motion_head_state": {k: v.detach().cpu() for k, v in motion_head.state_dict().items()},
            "scan_count": SCAN_COUNT_V18,
            "epoch": ep + 1,
            "all_scenes": bool(args.all_scenes),
            "val": val_info,
        }
        torch.save(payload, args.out_dir / f"houyi-v18-joint-12scan-e{ep+1:02d}.pt")
        if comb_metric >= best_score:
            best_score = comb_metric
            torch.save(payload, out_ckpt)
            print(f"  [SAVED BEST] {out_ckpt} (score={comb_metric:.5f})", flush=True)

    print(f"\n[DONE] Houyi-V18-Joint-12scan training finished in {(time.time()-t_start)/60:.1f}m!", flush=True)


if __name__ == "__main__":
    main()
