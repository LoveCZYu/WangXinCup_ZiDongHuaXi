"""Production B02 Master Tracker (Houyi-Dense + Formation-Aware Group Tracker).

Self-contained, zero external scratch dependencies, 100% causal scan-by-scan online tracker.
Key components:
1. Houyi v2 Transformer Point Classifier (8-scan causal window, bf16 SDPA, ~25ms/circle on RTX 5070).
2. Competition-Safe Level-0 Atomic Formation-Template Matching (k=2, k=3, and 2-of-3 sub-pairs).
3. Global Optimal Bipartite Association (scipy linear_sum_assignment) with 3D Cartesian kinematics,
   circular Azimuth wraparound, adaptive close-range Az gate, and heading consistency.
4. Parallel-Velocity co_obs>=1 Formation Locking + Zombie Duplicate Track Pruning.
5. Bounded Tight-Pair Partial-Detection Ambiguity Guard + Centroid & Wingman-Coupled Coasting.
6. Active-Missed + Dormant Bi-Directional 3D OLS Track Segment Association (Tier-1 & Tier-2 TSA).
7. Zone-Aware Safe 2-Point Early Confirmation & Low-Threshold Tentative Initiation.
"""
from collections import defaultdict, deque
import itertools
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

PHI_0_RAD = math.radians(0.23)
KAPPA_RAD = math.radians(5.73)
TWO_PI = 2.0 * math.pi
SCAN_COUNT = 8
SECTOR_BATCH = 4

DEFAULT_TRACKER_PARAMS = {
    "min_cand_score": 0.04,
    "gate_az": 0.028,
    "gate_el": 0.075,
    "gate_r": 0.022,
    "rescue_scale": 1.60,
    "rescue_long_scale": 1.95,
    "rescue_min_score": 0.50,
    "heading_pen_weight": 0.35,
    "age_bonus": 0.07,
    "confirm_hits": 3,
    "max_missed_conf": 8,
    "max_missed_tent": 2,
    "max_dormant": 30,
    "birth_score_min": 0.08,
    "early_birth_1_score": 0.975,
    "early_birth_2_score": 0.65,
    "safe_zone_mult": 2.8,
    "eb1_safe_mult": 2.8,
    "eb2_safe_mult": 0.0,
    "sibling_sep_az": 0.022,
    "sibling_sep_r": 0.018,
    "enter_score": 0.50,
    "maintain_score": 0.10,
    "use_atomic_l0": True,
    "l0_k3": True,
    "l0_k3_subpair": True,
    "l0_centroid_scale": 1.18,
    "l0_max_kin_err": 1.15,
    "l0_el_gate": 0.14,
    "l0_min_pt_score": 0.25,
    "use_formation": True,
    "parallel_co1_lock": True,
    "form_min_consistent": 2,
    "form_use_amp": True,
    "amp_sig_floor": 0.0025,
    "form_ambiguity_guard": True,
    "form_amb_az": 0.012,
    "form_amb_r": 0.009,
    "form_amb_margin": 99.0,
    "max_consec_guard": 5,
    "form_max_missed": 22,
    "zombie_prune_scans": 5,
    "use_tsa": True,
    "tsa_search_active_missed": True,
    "tsa_tier1_cos": 0.32,
    "tsa_tier2_cos": 0.22,
    "tsa_tier2_mid_dist": 0.20,
    "tsa_llr_th": 1.15,
    "r_cost_w": 0.40,
    "r_sigma": 0.005,
    "mc_add": 0.30,
    "filter_low_score_form": True,
    "link_conf_to_newborn": True,
    "l0_include_newborn": True,
    "allow_any_age_cn_co1": True,
    "aff_bipartite_w": 1.00,
    "aff_mc_add": 0.55,
    "aff_tent_w": 0.45,
    "aff_neutral": 0.35,
    "use_imm": True,
    "imm_strength": 1.0,
    "ctx_w": 0.25,
    "ctx_mode": "nll",
    "ctx_mc_add": 0.18,
    "ctx_all_edges": True,
    "perm_sep_daz": 0.022,
    "perm_sep_dr": 0.020,
    "adapt_form_sigmas": True,
    "form_outlier_reject": True,
    "form_outlier_min_cons": 3,
    "form_outlier_thr_az": 0.0045,
    "form_outlier_thr_el": 0.0070,
    "form_outlier_thr_r": 0.0045,
    "form_outlier_margin": 0.15,
}


def wrap_az(d_az: float) -> float:
    if d_az > 0.5:
        return d_az - 1.0
    if d_az < -0.5:
        return d_az + 1.0
    return d_az


def to_cart(az: float, el: float, r: float):
    az_rad = az * TWO_PI - math.pi
    el_rad = PHI_0_RAD + KAPPA_RAD * el
    cos_el = math.cos(el_rad)
    return (r * cos_el * math.sin(az_rad),
            r * cos_el * math.cos(az_rad),
            r * math.sin(el_rad))


def from_cart(x: float, y: float, z: float):
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-8:
        return 0.5, 0.0, 0.0
    el_rad = math.asin(max(-1.0, min(1.0, z / r)))
    az_rad = math.atan2(x, y)
    az = (az_rad + math.pi) / TWO_PI
    if az >= 1.0:
        az -= 1.0
    elif az < 0.0:
        az += 1.0
    el = (el_rad - PHI_0_RAD) / KAPPA_RAD
    return az, el, r


def _out(point, batch_id: int):
    return {
        "Time": int(point["Time"]),
        "Azimuth": point["Azimuth"],
        "Elevation": point["Elevation"],
        "Distance": point["Distance"],
        "BatchID": int(batch_id),
        "CircleID": int(point["CircleID"]),
        "PntID": int(point["PntID"]),
    }


class TrackState:
    __slots__ = (
        "batch", "az", "el", "raw_el", "r", "amp", "snr",
        "x", "y", "z", "vx", "vy", "vz",
        "last_scan", "real_last_scan", "birth_scan", "hits", "strong_hits",
        "emitted", "score_sum", "max_score", "last_score",
        "in_protected_birth", "amp_ema", "hist_cart",
        "omega", "mu_ct", "use_imm", "imm_strength",
    )

    def __init__(self, batch, p, score, scan, in_protected=False, use_imm=True, imm_strength=1.0):
        self.batch = batch
        self.az = p["Azimuth"]
        self.raw_el = p["Elevation"]
        self.el = p["Elevation"]
        self.r = p["Distance"]
        self.amp = p["Amp"]
        self.snr = p["SNR"]
        self.x, self.y, self.z = to_cart(self.az, self.el, self.r)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.last_scan = scan
        self.real_last_scan = scan
        self.birth_scan = scan
        self.hits = 1
        self.strong_hits = 1 if score >= 0.50 else 0
        self.emitted = False
        self.score_sum = score
        self.max_score = score
        self.last_score = score
        self.in_protected_birth = in_protected
        self.amp_ema = p["Amp"]
        self.hist_cart = deque(maxlen=8)
        self.hist_cart.append((scan, self.x, self.y, self.z, score))
        self.omega = 0.0
        self.mu_ct = 0.10
        self.use_imm = use_imm
        self.imm_strength = imm_strength

    def _predict_ct_xy(self, dt: float):
        w = self.omega
        if abs(w) < 1e-3:
            return self.x + self.vx * dt, self.y + self.vy * dt
        wd = w * dt
        if abs(wd) > 1.2:
            wd = math.copysign(1.2, wd)
            w = wd / max(1e-6, dt)
        s = math.sin(wd)
        c = math.cos(wd)
        dx = (s / w) * self.vx - ((1.0 - c) / w) * self.vy
        dy = ((1.0 - c) / w) * self.vx + (s / w) * self.vy
        return self.x + dx, self.y + dy

    def predict_sph(self, scan: int):
        dt = scan - self.last_scan
        if dt <= 0 or self.hits < 2:
            return self.az, self.el, self.r
        xcv = self.x + self.vx * dt
        ycv = self.y + self.vy * dt
        z = self.z + self.vz * dt
        if (not self.use_imm) or self.hits < 4 or self.imm_strength <= 0.0 or abs(self.omega) < 0.005 or self.mu_ct < 0.10:
            return from_cart(xcv, ycv, z)
        xct, yct = self._predict_ct_xy(float(dt))
        w_ct = min(0.95, self.mu_ct * self.imm_strength)
        return from_cart((1.0 - w_ct) * xcv + w_ct * xct, (1.0 - w_ct) * ycv + w_ct * yct, z)

    def robust_vel_cart(self):
        if len(self.hist_cart) < 2:
            return self.vx, self.vy, self.vz
        pts = [h for h in self.hist_cart if h[4] >= 0.35]
        if len(pts) < 2:
            pts = list(self.hist_cart)
        t0 = pts[0][0]
        ts = np.array([h[0] - t0 for h in pts], dtype=np.float64)
        t_mean = ts.mean()
        denom = float(np.sum((ts - t_mean) ** 2))
        if denom < 1e-6:
            return self.vx, self.vy, self.vz
        xs = np.array([h[1] for h in pts], dtype=np.float64)
        ys = np.array([h[2] for h in pts], dtype=np.float64)
        zs = np.array([h[3] for h in pts], dtype=np.float64)
        dt_c = ts - t_mean
        return (
            float(np.dot(dt_c, xs - xs.mean()) / denom),
            float(np.dot(dt_c, ys - ys.mean()) / denom),
            float(np.dot(dt_c, zs - zs.mean()) / denom),
        )

    def update(self, p, score, scan, vel_gain=0.35):
        dt = max(1, scan - self.last_scan)
        naz, raw_el, nr = p["Azimuth"], p["Elevation"], p["Distance"]
        if self.use_imm and self.hits >= 4 and dt <= 3:
            ix, iy, _ = to_cart(naz, self.el, nr)
            xcv = self.x + self.vx * dt
            ycv = self.y + self.vy * dt
            xct, yct = self._predict_ct_xy(float(dt))
            e_cv = (ix - xcv) ** 2 + (iy - ycv) ** 2
            e_ct = (ix - xct) ** 2 + (iy - yct) ** 2
            p_ct_pred = 0.06 * (1.0 - self.mu_ct) + 0.88 * self.mu_ct
            sig2 = (0.0045 ** 2) * dt
            ll_diff = max(-6.0, min(6.0, (e_cv - e_ct) / (2.0 * sig2)))
            num_ct = p_ct_pred * math.exp(ll_diff)
            self.mu_ct = max(0.05, min(0.95, num_ct / ((1.0 - p_ct_pred) + num_ct)))

        ovx, ovy = self.vx, self.vy
        self.raw_el = raw_el
        nx, ny, nz = to_cart(naz, raw_el, nr)
        g = 1.0 if self.hits == 1 else (0.6 if self.hits == 2 else vel_gain)
        trust = min(1.0, max(0.15, score / 0.50))
        eff_g = g * trust
        self.vx = (1.0 - eff_g) * self.vx + eff_g * ((nx - self.x) / dt)
        self.vy = (1.0 - eff_g) * self.vy + eff_g * ((ny - self.y) / dt)
        self.vz = (1.0 - eff_g) * self.vz + eff_g * ((nz - self.z) / dt)
        if self.use_imm:
            v1_sq = ovx * ovx + ovy * ovy
            v2_sq = self.vx * self.vx + self.vy * self.vy
            if self.hits >= 3 and dt <= 3 and v1_sq > 0.0035 ** 2 and v2_sq > 0.0035 ** 2:
                d_ang = math.atan2(ovx * self.vy - ovy * self.vx, ovx * self.vx + ovy * self.vy) / dt
                if abs(d_ang) < 0.35:
                    self.omega = 0.70 * self.omega + 0.30 * d_ang
                else:
                    self.omega *= 0.85
            elif dt > 3:
                self.omega *= 0.75
        self.x, self.y, self.z = nx, ny, nz
        self.az, self.el, self.r = naz, raw_el, nr
        self.amp = p["Amp"]
        self.snr = p["SNR"]
        self.amp_ema = 0.7 * self.amp_ema + 0.3 * p["Amp"]
        self.last_scan = scan
        self.real_last_scan = scan
        self.hits += 1
        if score >= 0.50:
            self.strong_hits = min(10, self.strong_hits + 1)
        self.score_sum += score
        if score > self.max_score:
            self.max_score = score
        self.last_score = score
        self.hist_cart.append((scan, nx, ny, nz, score))


class FormLink:
    __slots__ = (
        "d_az", "d_el", "d_r", "d_amp", "v_daz", "v_del", "v_dr",
        "co_obs", "consistent_obs", "last_co_scan", "amp_std_ema",
        "pos_step_ema", "consec_partial",
        "az_std_ema", "el_std_ema", "r_std_ema", "consec_split",
    )

    def __init__(self, t1: TrackState, t2: TrackState, scan: int):
        self.d_az = wrap_az(t1.az - t2.az)
        self.d_el = t1.raw_el - t2.raw_el
        self.d_r = t1.r - t2.r
        self.d_amp = t1.amp - t2.amp
        self.v_daz = 0.0
        self.v_del = 0.0
        self.v_dr = 0.0
        self.co_obs = 1
        self.consistent_obs = 1
        self.last_co_scan = scan
        self.amp_std_ema = 0.004
        self.pos_step_ema = 0.0015
        self.consec_partial = 0
        self.az_std_ema = 0.0008
        self.el_std_ema = 0.0006
        self.r_std_ema = 0.0007
        self.consec_split = 0

    def predict(self, scan: int):
        dt = min(5, max(0, scan - self.last_co_scan))
        return (self.d_az + self.v_daz * dt,
                self.d_el + self.v_del * dt,
                self.d_r + self.v_dr * dt,
                self.d_amp)

    def update(self, t1: TrackState, t2: TrackState, scan: int):
        if t1.last_score < 0.65 or t2.last_score < 0.65:
            return
        self.consec_partial = 0
        dt = max(1, scan - self.last_co_scan)
        new_daz = wrap_az(t1.az - t2.az)
        new_del = t1.raw_el - t2.raw_el
        new_dr = t1.r - t2.r
        new_damp = t1.amp - t2.amp

        pdaz, pdel, pdr, pdamp = self.predict(scan)
        e_az_abs = abs(new_daz - pdaz)
        e_el_abs = abs(new_del - pdel)
        e_r_abs = abs(new_dr - pdr)
        step_err = math.sqrt(e_az_abs * e_az_abs + e_el_abs * e_el_abs + e_r_abs * e_r_abs) / dt
        amp_err = abs(new_damp - pdamp)

        if self.consistent_obs >= 2:
            if step_err > 0.0045 or (self.amp_std_ema < 0.015 and amp_err > 0.035):
                return

        if step_err < 0.0030 and amp_err < 0.04:
            self.consistent_obs += 1
        else:
            self.consistent_obs = max(0, self.consistent_obs - 1)

        if self.co_obs >= 1:
            self.az_std_ema = 0.80 * self.az_std_ema + 0.20 * e_az_abs
            self.el_std_ema = 0.80 * self.el_std_ema + 0.20 * e_el_abs
            self.r_std_ema = 0.80 * self.r_std_ema + 0.20 * e_r_abs

        if self.co_obs >= 1 and dt <= 6 and step_err < 0.0040:
            av = 0.20
            self.v_daz = (1.0 - av) * self.v_daz + av * (new_daz - self.d_az) / dt
            self.v_del = (1.0 - av) * self.v_del + av * (new_del - self.d_el) / dt
            self.v_dr = (1.0 - av) * self.v_dr + av * (new_dr - self.d_r) / dt

        ap = 0.55 if self.consistent_obs < 4 else 0.35
        self.d_az = (1.0 - ap) * self.d_az + ap * new_daz
        self.d_el = (1.0 - ap) * self.d_el + ap * new_del
        self.d_r = (1.0 - ap) * self.d_r + ap * new_dr
        aa = 0.25 if self.consistent_obs < 4 else 0.05
        self.amp_std_ema = 0.80 * self.amp_std_ema + 0.20 * amp_err
        self.pos_step_ema = 0.80 * self.pos_step_ema + 0.20 * step_err
        self.d_amp = (1.0 - aa) * self.d_amp + aa * new_damp
        self.co_obs += 1
        self.last_co_scan = scan


class TrackLongMemory:
    """24-scan rolling identity & signal statistics per track (beyond 8-scan Houyi window)."""
    __slots__ = ("amps", "snrs", "els", "speeds", "matched_flags", "emitted_count")

    def __init__(self, p):
        self.amps = deque([float(p["Amp"])], maxlen=24)
        self.snrs = deque([float(p["SNR"])], maxlen=24)
        self.els = deque([float(p["Elevation"])], maxlen=24)
        self.speeds = deque(maxlen=20)
        self.matched_flags = deque([1], maxlen=24)
        self.emitted_count = 0

    def record_hit(self, p, speed_xy: float):
        self.amps.append(float(p["Amp"]))
        self.snrs.append(float(p["SNR"]))
        self.els.append(float(p["Elevation"]))
        if speed_xy > 0.0:
            self.speeds.append(speed_xy)
        self.matched_flags.append(1)

    def record_miss(self):
        self.matched_flags.append(0)

    def amp_z(self, amp: float) -> float:
        arr = self.amps
        n = len(arr)
        m = sum(arr) / n
        if n < 3:
            return (amp - m) / 0.12
        var = sum((x - m) ** 2 for x in arr) / n
        std = max(0.025, math.sqrt(var))
        return max(-6.0, min(6.0, (amp - m) / std))

    def snr_z(self, snr: float) -> float:
        arr = self.snrs
        n = len(arr)
        m = sum(arr) / n
        if n < 3:
            return (snr - m) / 0.12
        var = sum((x - m) ** 2 for x in arr) / n
        std = max(0.025, math.sqrt(var))
        return max(-6.0, min(6.0, (snr - m) / std))

    def miss_rate(self) -> float:
        n = len(self.matched_flags)
        return 1.0 - (sum(self.matched_flags) / n) if n else 0.0


class FastNumpyConflictNet:
    """Calibrated float32 NumPy ContextualConflictNet (~0.1 ms/scan CPU, exact PyTorch parity)."""

    def __init__(self, ckpt_path):
        ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        def npw(k):
            return sd[k].detach().cpu().numpy().astype(np.float32)
        self.in_w, self.in_b = npw("in_norm.weight"), npw("in_norm.bias")
        self.fc1_w, self.fc1_b = npw("fc1.weight").T, npw("fc1.bias")
        self.b1_ln1_w, self.b1_ln1_b = npw("block1.0.weight"), npw("block1.0.bias")
        self.b1_fc1_w, self.b1_fc1_b = npw("block1.3.weight").T, npw("block1.3.bias")
        self.b1_ln2_w, self.b1_ln2_b = npw("block1.4.weight"), npw("block1.4.bias")
        self.b1_fc2_w, self.b1_fc2_b = npw("block1.6.weight").T, npw("block1.6.bias")
        self.h_ln_w, self.h_ln_b = npw("head.0.weight"), npw("head.0.bias")
        self.h_fc1_w, self.h_fc1_b = npw("head.2.weight").T, npw("head.2.bias")
        self.h_fc2_w, self.h_fc2_b = npw("head.4.weight").T, npw("head.4.bias")
        self.temp = float(np.clip(sd["temp"].item(), 0.3, 5.0))

    @staticmethod
    def _ln(x, w, b):
        m = x.mean(axis=-1, keepdims=True)
        v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
        return (x - m) / np.sqrt(v + 1e-5) * w + b

    @staticmethod
    def _silu(x):
        return x / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def predict_probs(self, X: np.ndarray) -> np.ndarray:
        h = self._ln(X, self.in_w, self.in_b) @ self.fc1_w + self.fc1_b
        r = self._silu(self._ln(h, self.b1_ln1_w, self.b1_ln1_b)) @ self.b1_fc1_w + self.b1_fc1_b
        r = self._silu(self._ln(r, self.b1_ln2_w, self.b1_ln2_b)) @ self.b1_fc2_w + self.b1_fc2_b
        h = h + r
        u = self._silu(self._ln(h, self.h_ln_w, self.h_ln_b)) @ self.h_fc1_w + self.h_fc1_b
        logits = (self._silu(u) @ self.h_fc2_w + self.h_fc2_b).squeeze(-1) / self.temp
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _build_conflict_edge_features(tracker, memories: dict, track_ids, cand_pts, cand_scores,
                                  scan: int, gate_scale: float, occupied: set,
                                  min_pt_score: float = 0.0, long_scale=None):
    """Build 34-dim competitor-aware conflict graph features for gated (track, det) edges."""
    if not track_ids or not cand_pts:
        return []
    raw_edges = []
    by_track = {}
    by_det = {}
    form_co = {}
    form_sibs = {}
    for (b1, b2), lk in tracker.form_links.items():
        if lk.consistent_obs >= 1:
            if lk.consistent_obs > form_co.get(b1, 0):
                form_co[b1] = lk.consistent_obs
            if lk.consistent_obs > form_co.get(b2, 0):
                form_co[b2] = lk.consistent_obs
            form_sibs.setdefault(b1, set()).add(b2)
            form_sibs.setdefault(b2, set()).add(b1)

    for bid in track_ids:
        t = tracker.tracks[bid]
        dt = scan - t.last_scan
        real_dt = scan - t.real_last_scan
        paz, pel, pr = t.predict_sph(scan)
        if t.hits == 1:
            eff_scale = max(gate_scale, 1.75)
        elif long_scale is not None and dt >= 3:
            eff_scale = long_scale
        else:
            eff_scale = gate_scale
        rad = min(1.0 + 0.22 * max(0, dt - 1), 2.15) * eff_scale
        rad_sq = rad * rad
        g_az = max(tracker.gate_az, 0.008 / max(pr, 0.08))
        v_xy = math.hypot(t.vx, t.vy)
        mem = memories.get(bid)
        t_form_co = min(1.0, form_co.get(bid, 0) / 20.0)
        in_form = 1.0 if bid in form_sibs else 0.0
        rkeys = tracker.track_recent_keys.get(bid)

        for ci, p in enumerate(cand_pts):
            if ci in occupied or cand_scores[ci] < min_pt_score:
                continue
            if abs(p["Distance"] - pr) > tracker.gate_r * rad * 1.15:
                continue
            d_sq = tracker._norm_dist_sq(paz, pel, pr, p["Azimuth"], p["Elevation"], p["Distance"], pr)
            if d_sq > rad_sq or abs(p["Elevation"] - pel) > 0.25:
                continue
            d_az_n = abs(wrap_az(p["Azimuth"] - paz)) / (g_az * rad)
            d_el_n = abs(p["Elevation"] - pel) / (tracker.gate_el * rad)
            d_r_n = abs(p["Distance"] - pr) / (tracker.gate_r * rad)
            dr_fine = min(3.0, abs(p["Distance"] - pr) / (0.005 * rad))
            dsq_n = d_sq / rad_sq

            cos_h = 1.0
            spd_rat = 1.0
            if t.hits >= 4 and v_xy > 0.004:
                cx, cy, _ = to_cart(p["Azimuth"], t.el, p["Distance"])
                dx, dy = (cx - t.x) / max(1, dt), (cy - t.y) / max(1, dt)
                d_xy = math.hypot(dx, dy)
                if d_xy > 0.003:
                    cos_h = (t.vx * dx + t.vy * dy) / (v_xy * d_xy)
                    spd_rat = min(3.0, d_xy / v_xy)

            p_aff = tracker._get_pair_aff(bid, ci)
            has_aff = 1.0 if p_aff is not None else 0.0
            p_aff_val = p_aff if p_aff is not None else 0.5
            p_aff_max = p_aff_val
            p_aff_last1 = p_aff_val
            if rkeys and tracker._cur_aff:
                orig_i = tracker._cand_orig_idx[ci]
                row = tracker._cur_aff.get(orig_i)
                if row:
                    vals = [row[k] for k in rkeys if k in row]
                    if vals:
                        p_aff_max = max(vals)
                        p_aff_last1 = vals[-1]

            amp_z = abs(mem.amp_z(p["Amp"])) if mem is not None else 0.0
            snr_z = abs(mem.snr_z(p["SNR"])) if mem is not None else 0.0
            miss_r = mem.miss_rate() if mem is not None else 0.0

            e_idx = len(raw_edges)
            raw_edges.append({
                "bid": bid, "ci": ci,
                "dt": min(10.0, float(dt)) / 10.0,
                "real_dt": min(15.0, float(real_dt)) / 15.0,
                "d_az_n": d_az_n, "d_el_n": d_el_n, "d_r_n": d_r_n,
                "dsq_n": dsq_n, "dr_fine": dr_fine,
                "cos_h": cos_h, "spd_rat": spd_rat,
                "mu_ct": float(getattr(t, "mu_ct", 0.1)),
                "omega": min(0.35, abs(float(getattr(t, "omega", 0.0)))) / 0.35,
                "sc": float(cand_scores[ci]),
                "p_aff": p_aff_val, "has_aff": has_aff,
                "p_aff_max": p_aff_max, "p_aff_last1": p_aff_last1,
                "hits_log": min(1.0, math.log1p(t.hits) / 5.5),
                "strong_n": min(1.0, t.strong_hits / 10.0),
                "amp_z": amp_z / 6.0, "snr_z": snr_z / 6.0,
                "miss_r": miss_r, "in_form": in_form, "form_co": t_form_co,
            })
            by_track.setdefault(bid, []).append(e_idx)
            by_det.setdefault(ci, []).append(e_idx)

    if not raw_edges:
        return []

    out = []
    for e_idx, e in enumerate(raw_edges):
        bid = e["bid"]
        ci = e["ci"]
        t_list = by_track[bid]
        d_list = by_det[ci]
        n_ct = len(d_list)
        n_cd = len(t_list)
        is_conflict = (n_ct >= 2 or n_cd >= 2)

        bo_dsq, bo_dr, bo_paff, bo_amp, bo_is_sib = 2.0, 3.0, 0.0, 1.0, 0.0
        if n_ct >= 2:
            sibs = form_sibs.get(bid, ())
            best_score = 1e9
            for oi in d_list:
                if oi == e_idx:
                    continue
                oe = raw_edges[oi]
                cs = oe["dsq_n"] + 0.4 * min(1.0, oe["dr_fine"] ** 2) + (1.0 - oe["p_aff"])
                if cs < best_score:
                    best_score = cs
                    bo_dsq, bo_dr, bo_paff, bo_amp = oe["dsq_n"], oe["dr_fine"], oe["p_aff"], oe["amp_z"]
                    bo_is_sib = 1.0 if oe["bid"] in sibs else 0.0

        bd_dsq, bd_paff = 2.0, 0.0
        if n_cd >= 2:
            best_score = 1e9
            for oi in t_list:
                if oi == e_idx:
                    continue
                oe = raw_edges[oi]
                cs = oe["dsq_n"] + 0.4 * min(1.0, oe["dr_fine"] ** 2) + (1.0 - oe["p_aff"])
                if cs < best_score:
                    best_score = cs
                    bd_dsq, bd_paff = oe["dsq_n"], oe["p_aff"]

        feat = np.array([
            e["dt"], e["real_dt"], e["d_az_n"], e["d_el_n"], e["d_r_n"],
            e["dsq_n"], e["dr_fine"], e["cos_h"], e["spd_rat"], e["mu_ct"], e["omega"],
            e["sc"], e["p_aff"], e["has_aff"], e["p_aff_max"], e["p_aff_last1"],
            e["hits_log"], e["strong_n"], e["amp_z"], e["snr_z"], e["miss_r"], e["in_form"], e["form_co"],
            min(1.0, (n_ct - 1) / 4.0), min(1.0, (n_cd - 1) / 4.0),
            bo_dsq, max(-2.0, min(2.0, bo_dsq - e["dsq_n"])), max(-3.0, min(3.0, bo_dr - e["dr_fine"])),
            bo_paff, e["p_aff"] - bo_paff, max(-1.0, min(1.0, bo_amp - e["amp_z"])),
            bo_is_sib, max(-2.0, min(2.0, bd_dsq - e["dsq_n"])), e["p_aff"] - bd_paff,
        ], dtype=np.float32)
        out.append((bid, ci, feat, is_conflict))
    return out


SEQ_LEN = 24
SEQ_DIM = 12
EDGE_DIM_V18 = 40
MAX_COMP_EDGES = 24


class TrackSeqMemory:
    """24-scan rolling trajectory & signal sequence memory per track (V18-C)."""
    __slots__ = ("amps", "snrs", "els", "matched_flags", "seq_rows")

    def __init__(self, p, score: float, scan: int):
        self.amps = deque([float(p["Amp"])], maxlen=SEQ_LEN)
        self.snrs = deque([float(p["SNR"])], maxlen=SEQ_LEN)
        self.els = deque([float(p["Elevation"])], maxlen=SEQ_LEN)
        self.matched_flags = deque([1], maxlen=SEQ_LEN)
        self.seq_rows = deque([(
            scan, float(p["Azimuth"]), float(p["Elevation"]), float(p["Distance"]),
            0.0, 0.0, 0.0, 0.1, float(p["Amp"]), float(p["SNR"]), float(score), 1.0,
        )], maxlen=SEQ_LEN)

    def record_hit(self, t, p, score: float, scan: int):
        self.amps.append(float(p["Amp"]))
        self.snrs.append(float(p["SNR"]))
        self.els.append(float(p["Elevation"]))
        self.matched_flags.append(1)
        self.seq_rows.append((
            scan, float(p["Azimuth"]), float(p["Elevation"]), float(p["Distance"]),
            float(t.vx), float(t.vy), float(getattr(t, "omega", 0.0)), float(getattr(t, "mu_ct", 0.1)),
            float(p["Amp"]), float(p["SNR"]), float(score), 1.0,
        ))

    def record_miss(self, t, scan: int):
        self.matched_flags.append(0)
        self.seq_rows.append((
            scan, float(t.az), float(t.el), float(t.r),
            float(t.vx), float(t.vy), float(getattr(t, "omega", 0.0)), float(getattr(t, "mu_ct", 0.1)),
            float(t.amp), float(t.snr), 0.0, 0.0,
        ))

    def amp_z(self, amp: float) -> float:
        arr = self.amps
        n = len(arr)
        m = sum(arr) / n
        if n < 3:
            return (amp - m) / 0.12
        var = sum((x - m) ** 2 for x in arr) / n
        return max(-6.0, min(6.0, (amp - m) / max(0.025, math.sqrt(var))))

    def snr_z(self, snr: float) -> float:
        arr = self.snrs
        n = len(arr)
        m = sum(arr) / n
        if n < 3:
            return (snr - m) / 0.12
        var = sum((x - m) ** 2 for x in arr) / n
        return max(-6.0, min(6.0, (snr - m) / max(0.025, math.sqrt(var))))

    def miss_rate(self) -> float:
        n = len(self.matched_flags)
        return 1.0 - (sum(self.matched_flags) / n) if n else 0.0

    def to_seq_tensor(self, scan: int, ref_az: float, ref_el: float, ref_r: float) -> np.ndarray:
        """Build right-aligned (SEQ_LEN, SEQ_DIM) float32 sequence relative to current predicted state."""
        out = np.zeros((SEQ_LEN, SEQ_DIM), dtype=np.float32)
        rows = self.seq_rows
        n = len(rows)
        off = SEQ_LEN - n
        for k, (sc, az, el, r, vx, vy, om, mu, amp, snr, sc_p, is_r) in enumerate(rows):
            dt = min(30.0, float(max(0, scan - sc))) / 24.0
            daz = max(-5.0, min(5.0, wrap_az(az - ref_az) * 35.0))
            del_ = max(-5.0, min(5.0, (el - ref_el) * 12.0))
            dr = max(-5.0, min(5.0, (r - ref_r) * 50.0))
            out[off + k] = (
                dt, daz, del_, dr,
                max(-5.0, min(5.0, vx * 40.0)),
                max(-5.0, min(5.0, vy * 40.0)),
                max(-3.0, min(3.0, om * 5.0)),
                mu, amp, snr, sc_p, is_r,
            )
        return out


def _build_rel_mat(bids, cis) -> np.ndarray:
    e = len(bids)
    rel = np.zeros((e, e), dtype=np.int8)
    for i in range(e):
        rel[i, i] = 3
        for j in range(i + 1, e):
            if bids[i] == bids[j]:
                rel[i, j] = rel[j, i] = 1
            elif cis[i] == cis[j]:
                rel[i, j] = rel[j, i] = 2
    return rel


def _build_v18_conflict_graph(tracker, memories: dict, track_ids, cand_pts, cand_scores,
                              scan: int, gate_scale: float, occupied: set,
                              cur_motion=None, min_pt_score: float = 0.0, long_scale=None):
    """Build 40-dim competitor+motion edges, 24-scan track sequences, and conflict components."""
    if not track_ids or not cand_pts:
        return [], {}, []

    raw_edges = []
    by_track = {}
    by_det = {}
    form_co = {}
    form_sibs = {}
    for (b1, b2), lk in tracker.form_links.items():
        if lk.consistent_obs >= 1:
            if lk.consistent_obs > form_co.get(b1, 0):
                form_co[b1] = lk.consistent_obs
            if lk.consistent_obs > form_co.get(b2, 0):
                form_co[b2] = lk.consistent_obs
            form_sibs.setdefault(b1, set()).add(b2)
            form_sibs.setdefault(b2, set()).add(b1)

    track_seqs = {}
    for bid in track_ids:
        t = tracker.tracks[bid]
        dt = scan - t.last_scan
        real_dt = scan - t.real_last_scan
        paz, pel, pr = t.predict_sph(scan)
        if t.hits == 1:
            eff_scale = max(gate_scale, 1.75)
        elif long_scale is not None and dt >= 3:
            eff_scale = long_scale
        else:
            eff_scale = gate_scale
        rad = min(1.0 + 0.22 * max(0, dt - 1), 2.15) * eff_scale
        rad_sq = rad * rad
        g_az = max(tracker.gate_az, 0.008 / max(pr, 0.08))
        v_xy = math.hypot(t.vx, t.vy)
        mem = memories.get(bid)
        t_form_co = min(1.0, form_co.get(bid, 0) / 20.0)
        in_form = 1.0 if bid in form_sibs else 0.0
        rkeys = tracker.track_recent_keys.get(bid)

        for ci, p in enumerate(cand_pts):
            if ci in occupied or cand_scores[ci] < min_pt_score:
                continue
            if abs(p["Distance"] - pr) > tracker.gate_r * rad * 1.15:
                continue
            d_sq = tracker._norm_dist_sq(paz, pel, pr, p["Azimuth"], p["Elevation"], p["Distance"], pr)
            if d_sq > rad_sq or abs(p["Elevation"] - pel) > 0.25:
                continue

            if bid not in track_seqs:
                if mem is not None:
                    track_seqs[bid] = mem.to_seq_tensor(scan, paz, pel, pr)
                else:
                    track_seqs[bid] = np.zeros((SEQ_LEN, SEQ_DIM), dtype=np.float32)

            d_az_n = abs(wrap_az(p["Azimuth"] - paz)) / (g_az * rad)
            d_el_n = abs(p["Elevation"] - pel) / (tracker.gate_el * rad)
            d_r_n = abs(p["Distance"] - pr) / (tracker.gate_r * rad)
            dr_fine = min(3.0, abs(p["Distance"] - pr) / (0.005 * rad))
            dsq_n = d_sq / rad_sq

            cos_h = 1.0
            spd_rat = 1.0
            if t.hits >= 4 and v_xy > 0.004:
                cx, cy, _ = to_cart(p["Azimuth"], t.el, p["Distance"])
                dx, dy = (cx - t.x) / max(1, dt), (cy - t.y) / max(1, dt)
                d_xy = math.hypot(dx, dy)
                if d_xy > 0.003:
                    cos_h = (t.vx * dx + t.vy * dy) / (v_xy * d_xy)
                    spd_rat = min(3.0, d_xy / v_xy)

            p_aff = tracker._get_pair_aff(bid, ci)
            has_aff = 1.0 if p_aff is not None else 0.0
            p_aff_val = p_aff if p_aff is not None else 0.5
            p_aff_max = p_aff_val
            p_aff_last1 = p_aff_val
            orig_i = tracker._cand_orig_idx[ci]
            if rkeys and tracker._cur_aff:
                row = tracker._cur_aff.get(orig_i)
                if row:
                    vals = [row[k] for k in rkeys if k in row]
                    if vals:
                        p_aff_max = max(vals)
                        p_aff_last1 = vals[-1]

            has_mot = 0.0
            b_daz_n, b_del_n, b_dr_n, b_dsq = 1.0, 1.0, 1.0, 1.5
            if cur_motion is not None and orig_i < len(cur_motion) and 1 <= real_dt <= 4:
                mv = cur_motion[orig_i]
                vaz_pred = float(mv[0]) / 50.0
                vel_pred = float(mv[1]) / 15.0
                vr_pred = float(mv[2]) / 80.0
                pred_prev_az = (p["Azimuth"] - vaz_pred * real_dt) % 1.0
                pred_prev_el = p["Elevation"] - vel_pred * real_dt
                pred_prev_r = p["Distance"] - vr_pred * real_dt
                b_daz_n = min(3.0, abs(wrap_az(pred_prev_az - t.az)) / (0.006 * max(1, real_dt)))
                b_del_n = min(3.0, abs(pred_prev_el - t.raw_el) / (0.025 * max(1, real_dt)))
                b_dr_n = min(3.0, abs(pred_prev_r - t.r) / (0.005 * max(1, real_dt)))
                b_dsq = min(4.0, 0.45 * b_daz_n * b_daz_n + 0.15 * b_del_n * b_del_n + 0.40 * b_dr_n * b_dr_n)
                has_mot = 1.0

            amp_z = abs(mem.amp_z(p["Amp"])) if mem is not None else 0.0
            snr_z = abs(mem.snr_z(p["SNR"])) if mem is not None else 0.0
            miss_r = mem.miss_rate() if mem is not None else 0.0

            e_idx = len(raw_edges)
            raw_edges.append({
                "bid": bid, "ci": ci,
                "dt": min(10.0, float(dt)) / 10.0,
                "real_dt": min(15.0, float(real_dt)) / 15.0,
                "d_az_n": d_az_n, "d_el_n": d_el_n, "d_r_n": d_r_n,
                "dsq_n": dsq_n, "dr_fine": dr_fine,
                "cos_h": cos_h, "spd_rat": spd_rat,
                "mu_ct": float(getattr(t, "mu_ct", 0.1)),
                "omega": min(0.35, abs(float(getattr(t, "omega", 0.0)))) / 0.35,
                "sc": float(cand_scores[ci]),
                "p_aff": p_aff_val, "has_aff": has_aff,
                "p_aff_max": p_aff_max, "p_aff_last1": p_aff_last1,
                "hits_log": min(1.0, math.log1p(t.hits) / 5.5),
                "strong_n": min(1.0, t.strong_hits / 10.0),
                "amp_z": amp_z / 6.0, "snr_z": snr_z / 6.0,
                "miss_r": miss_r, "in_form": in_form, "form_co": t_form_co,
                "has_mot": has_mot, "b_daz_n": b_daz_n, "b_del_n": b_del_n, "b_dr_n": b_dr_n, "b_dsq": b_dsq,
            })
            by_track.setdefault(bid, []).append(e_idx)
            by_det.setdefault(ci, []).append(e_idx)

    if not raw_edges:
        return [], {}, []

    edges = []
    for e_idx, e in enumerate(raw_edges):
        bid = e["bid"]
        ci = e["ci"]
        t_list = by_track[bid]
        d_list = by_det[ci]
        n_ct, n_cd = len(d_list), len(t_list)
        is_conflict = (n_ct >= 2 or n_cd >= 2)

        bo_dsq, bo_dr, bo_paff, bo_amp, bo_is_sib, bo_bdsq = 2.0, 3.0, 0.0, 1.0, 0.0, 3.0
        if n_ct >= 2:
            sibs = form_sibs.get(bid, ())
            best_s = 1e9
            for oi in d_list:
                if oi == e_idx:
                    continue
                oe = raw_edges[oi]
                cs = oe["dsq_n"] + 0.4 * min(1.0, oe["dr_fine"] ** 2) + (1.0 - oe["p_aff"])
                if cs < best_s:
                    best_s = cs
                    bo_dsq, bo_dr, bo_paff, bo_amp = oe["dsq_n"], oe["dr_fine"], oe["p_aff"], oe["amp_z"]
                    bo_is_sib = 1.0 if oe["bid"] in sibs else 0.0
                    bo_bdsq = oe["b_dsq"]

        bd_dsq, bd_paff = 2.0, 0.0
        if n_cd >= 2:
            best_s = 1e9
            for oi in t_list:
                if oi == e_idx:
                    continue
                oe = raw_edges[oi]
                cs = oe["dsq_n"] + 0.4 * min(1.0, oe["dr_fine"] ** 2) + (1.0 - oe["p_aff"])
                if cs < best_s:
                    best_s = cs
                    bd_dsq, bd_paff = oe["dsq_n"], oe["p_aff"]

        feat = np.array([
            e["dt"], e["real_dt"], e["d_az_n"], e["d_el_n"], e["d_r_n"],
            e["dsq_n"], e["dr_fine"], e["cos_h"], e["spd_rat"], e["mu_ct"], e["omega"],
            e["sc"], e["p_aff"], e["has_aff"], e["p_aff_max"], e["p_aff_last1"],
            e["hits_log"], e["strong_n"], e["amp_z"], e["snr_z"], e["miss_r"], e["in_form"], e["form_co"],
            min(1.0, (n_ct - 1) / 4.0), min(1.0, (n_cd - 1) / 4.0),
            bo_dsq, max(-2.0, min(2.0, bo_dsq - e["dsq_n"])), max(-3.0, min(3.0, bo_dr - e["dr_fine"])),
            bo_paff, e["p_aff"] - bo_paff, max(-1.0, min(1.0, bo_amp - e["amp_z"])),
            bo_is_sib, max(-2.0, min(2.0, bd_dsq - e["dsq_n"])), e["p_aff"] - bd_paff,
            e["has_mot"], e["b_daz_n"], e["b_del_n"], e["b_dr_n"], e["b_dsq"],
            max(-3.0, min(3.0, bo_bdsq - e["b_dsq"])),
        ], dtype=np.float32)
        edges.append((bid, ci, feat, is_conflict))

    n_e = len(edges)
    visited = [False] * n_e
    components = []
    for i in range(n_e):
        if visited[i]:
            continue
        comp = []
        stack = [i]
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            b_u, c_u = edges[u][0], edges[u][1]
            for v in by_track[b_u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
            for v in by_det[c_u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        components.append(comp)

    return edges, track_seqs, components


class BipartiteEdgeAttnLayer(nn.Module):
    """Multi-head edge self-attention with relation bias (peer/same_track/same_det/self)."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.10):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rel_bias = nn.Embedding(4, n_heads)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor, rel_mat: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        bsz, ne, d_m = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(bsz, ne, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + self.rel_bias(rel_mat).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~pad_mask[:, None, None, :], -1e4)
        attn = self.drop(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, ne, d_m)
        x = x + self.drop(self.out_proj(ctx))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class TrackSeqCrossAttnNet(nn.Module):
    """V18-C calibrated sequence + bipartite conflict-component attention associator."""

    def __init__(self, edge_dim: int = EDGE_DIM_V18, seq_dim: int = SEQ_DIM,
                 trk_dim: int = 64, d_model: int = 128, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.seq_in = nn.Sequential(nn.LayerNorm(seq_dim), nn.Linear(seq_dim, trk_dim), nn.SiLU())
        self.gru = nn.GRU(trk_dim, trk_dim, num_layers=1, batch_first=True)
        self.edge_in = nn.Sequential(nn.LayerNorm(edge_dim), nn.Linear(edge_dim, d_model), nn.SiLU())
        self.fuse = nn.Sequential(
            nn.Linear(d_model + trk_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList([BipartiteEdgeAttnLayer(d_model, n_heads) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.SiLU(), nn.Linear(d_model, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.temp = nn.Parameter(torch.tensor(1.0))

    def encode_tracks(self, seqs: torch.Tensor) -> torch.Tensor:
        _, hn = self.gru(self.seq_in(seqs))
        return hn[-1]

    def forward(self, edge_feats: torch.Tensor, edge_seqs: torch.Tensor,
                rel_mat: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        bsz, ne, slen, sdim = edge_seqs.shape
        h_trk = self.encode_tracks(edge_seqs.view(bsz * ne, slen, sdim)).view(bsz, ne, -1)
        h_e = self.edge_in(edge_feats)
        x = self.fuse(torch.cat((h_e, h_trk), dim=-1))
        for layer in self.layers:
            x = layer(x, rel_mat, pad_mask)
        return self.head(x).squeeze(-1) / self.temp.clamp(0.3, 5.0)


class FastSeqCrossAttnScorer:
    """Fast GPU/CPU inference wrapper for TrackSeqCrossAttnNet (~0.4 ms/scan)."""

    def __init__(self, ckpt_path, device: str = "cuda"):
        ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.model = TrackSeqCrossAttnNet(
            edge_dim=ckpt.get("edge_dim", EDGE_DIM_V18),
            seq_dim=ckpt.get("seq_dim", SEQ_DIM),
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def score_graph(self, edges, track_seqs, components, score_all: bool = True):
        if not edges or not components:
            return {}, set()
        sel_comps = [c for c in components if score_all or len(c) >= 2]
        if not sel_comps:
            return {}, set()
        capped = [
            c if len(c) <= MAX_COMP_EDGES else sorted(c, key=lambda ei: edges[ei][2][5])[:MAX_COMP_EDGES]
            for c in sel_comps
        ]
        bsz = len(capped)
        max_e = max(len(c) for c in capped)
        ef = np.zeros((bsz, max_e, EDGE_DIM_V18), dtype=np.float32)
        es = np.zeros((bsz, max_e, SEQ_LEN, SEQ_DIM), dtype=np.float32)
        rm = np.zeros((bsz, max_e, max_e), dtype=np.int64)
        pm = np.zeros((bsz, max_e), dtype=bool)
        for b, comp in enumerate(capped):
            bids = [edges[ei][0] for ei in comp]
            cis = [edges[ei][1] for ei in comp]
            ne = len(comp)
            for k, ei in enumerate(comp):
                ef[b, k] = edges[ei][2]
                es[b, k] = track_seqs[bids[k]]
            rm[b, :ne, :ne] = _build_rel_mat(bids, cis)
            pm[b, :ne] = True
        logits = self.model(
            torch.from_numpy(ef).to(self.device),
            torch.from_numpy(es).to(self.device),
            torch.from_numpy(rm).to(self.device),
            torch.from_numpy(pm).to(self.device),
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        pmap, cset = {}, set()
        for b, comp in enumerate(capped):
            is_c = len(comp) >= 2
            for k, ei in enumerate(comp):
                bid, ci, _, _ = edges[ei]
                pmap[(bid, ci)] = float(probs[b, k])
                if is_c:
                    cset.add((bid, ci))
        return pmap, cset


class MasterTracker:
    def __init__(self, conflict_net: FastNumpyConflictNet = None, seq_scorer: FastSeqCrossAttnScorer = None, **overrides):
        params = {**DEFAULT_TRACKER_PARAMS, **overrides}
        for k, v in params.items():
            setattr(self, k, v)
        self.conflict_net = conflict_net
        self.seq_scorer = seq_scorer
        self.scan = 0
        self.next_id = 1
        self.tracks: dict[int, TrackState] = {}
        self.dormant: dict[int, TrackState] = {}
        self.form_links: dict[tuple[int, int], FormLink] = {}
        self.group_full_obs: dict[tuple[int, ...], int] = {}
        self.track_recent_keys: dict[int, deque] = {}
        self.memories: dict[int, TrackLongMemory] = {}
        self.seq_memories: dict[int, TrackSeqMemory] = {}
        self._eb2_safe: dict[int, bool] = {}
        self._cur_aff: dict = {}
        self._cur_motion = None
        self._cand_orig_idx: list[int] = []

    def _compute_ctx_scores(self, track_ids, cand_pts, cand_scores, scan, gate_scale, occupied,
                            min_pt_score=0.0, long_scale=None):
        if self.ctx_w <= 0.0 or not track_ids or not cand_pts:
            return {}, set()
        if self.seq_scorer is not None:
            edges, track_seqs, comps = _build_v18_conflict_graph(
                self, self.seq_memories, track_ids, cand_pts, cand_scores, scan,
                gate_scale=gate_scale, occupied=occupied, cur_motion=self._cur_motion,
                min_pt_score=min_pt_score, long_scale=long_scale,
            )
            return self.seq_scorer.score_graph(edges, track_seqs, comps, score_all=self.ctx_all_edges)
        if self.conflict_net is None:
            return {}, set()
        edge_feats = _build_conflict_edge_features(
            self, self.memories, track_ids, cand_pts, cand_scores, scan,
            gate_scale=gate_scale, occupied=occupied, min_pt_score=min_pt_score, long_scale=long_scale,
        )
        if not edge_feats:
            return {}, set()
        sel = edge_feats if self.ctx_all_edges else [ef for ef in edge_feats if ef[3]]
        if not sel:
            return {}, set()
        X = np.stack([ef[2] for ef in sel], axis=0)
        probs = self.conflict_net.predict_probs(X)
        pmap, cset = {}, set()
        for (bid, ci, _, is_conf), pr in zip(sel, probs):
            pmap[(bid, ci)] = float(pr)
            if is_conf:
                cset.add((bid, ci))
        return pmap, cset

    def active_hist_keys(self) -> set[tuple[int, int]]:
        keys = set()
        for dq in self.track_recent_keys.values():
            keys.update(dq)
        return keys

    def _get_pair_aff(self, bid: int, cand_ci: int):
        keys = self.track_recent_keys.get(bid)
        if not keys or not self._cur_aff:
            return None
        orig_i = self._cand_orig_idx[cand_ci]
        row = self._cur_aff.get(orig_i)
        if not row:
            return None
        w_sum = 0.0
        p_sum = 0.0
        w = 1.0
        for hc, hp in reversed(keys):
            val = row.get((hc, hp))
            if val is not None:
                p_sum += w * val
                w_sum += w
            w *= 0.65
        if w_sum <= 0.0:
            return None
        return p_sum / w_sum

    def _norm_dist_sq(self, az1, el1, r1, az2, el2, r2, r_ref):
        g_az = max(self.gate_az, 0.008 / max(r_ref, 0.08))
        d_az = wrap_az(az1 - az2) / g_az
        d_el = (el1 - el2) / self.gate_el
        d_r = (r1 - r2) / self.gate_r
        el_term = min(0.65, d_el * d_el)
        return d_az * d_az + el_term + d_r * d_r

    def _is_confirmed(self, t: TrackState) -> bool:
        return t.emitted or t.hits >= self.confirm_hits

    def _is_parallel_co1(self, b1: int, b2: int, lk: FormLink) -> bool:
        if lk.co_obs < 1 or lk.consistent_obs < 1:
            return False
        if abs(lk.d_az) > 0.011 or abs(lk.d_r) > 0.009 or abs(lk.d_el) > 0.020:
            return False
        t1, t2 = self.tracks[b1], self.tracks[b2]
        if abs(t1.birth_scan - t2.birth_scan) <= 2 and t1.hits <= 5 and t2.hits <= 5:
            return True
        if self.allow_any_age_cn_co1:
            if (t1.hits <= 2 and self._is_confirmed(t2)) or (t2.hits <= 2 and self._is_confirmed(t1)):
                return True
        if t1.hits >= 2 and t2.hits >= 2:
            v1_sq = t1.vx * t1.vx + t1.vy * t1.vy
            v2_sq = t2.vx * t2.vx + t2.vy * t2.vy
            dv_sq = (t1.vx - t2.vx) ** 2 + (t1.vy - t2.vy) ** 2
            if dv_sq < 0.0035 * 0.0035:
                if v1_sq > 0.003 * 0.003 and v2_sq > 0.003 * 0.003:
                    cos_v = (t1.vx * t2.vx + t1.vy * t2.vy) / math.sqrt(v1_sq * v2_sq)
                    return cos_v >= 0.75
                return True
        return False

    def _link_ready(self, b1: int, b2: int, lk: FormLink, allow_co1: bool = False) -> bool:
        if lk.consistent_obs >= self.form_min_consistent:
            return True
        if allow_co1 and self.parallel_co1_lock and self._is_parallel_co1(b1, b2, lk):
            return True
        return False

    def _active_formation_groups(self, track_ids, allow_co1=False):
        adj = defaultdict(list)
        t_set = set(track_ids)
        for (b1, b2), lk in self.form_links.items():
            if b1 not in t_set or b2 not in t_set:
                continue
            if self._link_ready(b1, b2, lk, allow_co1=allow_co1):
                adj[b1].append(b2)
                adj[b2].append(b1)
        visited = set()
        groups = []
        for b in adj:
            if b in visited:
                continue
            comp = []
            stack = [b]
            visited.add(b)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            if len(comp) >= 2:
                comp.sort()
                groups.append(comp)
        return groups

    def _perm_sigmas(self, lk: FormLink):
        if not self.adapt_form_sigmas:
            return 0.0015, 0.0005, 0.0012
        sig_az = max(0.0015, min(0.0035, getattr(lk, "az_std_ema", 0.0008) * 3.0))
        sig_el = max(0.0005, min(0.0030, getattr(lk, "el_std_ema", 0.0006) * 3.0))
        sig_r = max(0.0012, min(0.0030, getattr(lk, "r_std_ema", 0.0007) * 3.0))
        return sig_az, sig_el, sig_r

    def _pair_template_err(self, p1, p2, pdaz, pdel, pdr, pdamp, astd, lk=None):
        da = abs(wrap_az((p1["Azimuth"] - p2["Azimuth"]) - pdaz))
        de = abs((p1["Elevation"] - p2["Elevation"]) - pdel)
        dr = abs((p1["Distance"] - p2["Distance"]) - pdr)
        if self.adapt_form_sigmas and lk is not None:
            sig_az = max(0.0014, min(0.0035, getattr(lk, "az_std_ema", 0.0008) * 3.0))
            sig_el = max(0.0005, min(0.0030, getattr(lk, "el_std_ema", 0.0006) * 3.0))
            sig_r = max(0.0010, min(0.0030, getattr(lk, "r_std_ema", 0.0007) * 3.0))
            if da > max(0.0042, sig_az * 3.5) or de > max(0.0036, sig_el * 4.5) or dr > max(0.0040, sig_r * 3.5):
                return None
            e_az = da / sig_az
            e_el = de / sig_el
            e_r = dr / sig_r
            err = min(40.0, e_az * e_az) + min(80.0, e_el * e_el) + min(40.0, e_r * e_r)
        else:
            if da > 0.0042 or de > 0.0036 or dr > 0.0040:
                return None
            e_az = da / 0.0014
            e_el = de / 0.0005
            e_r = dr / 0.0010
            err = e_az * e_az + e_el * e_el + e_r * e_r
        if self.form_use_amp:
            damp = abs((p1["Amp"] - p2["Amp"]) - pdamp)
            if astd < 0.012 and damp > 0.035:
                return None
            sig_a = max(self.amp_sig_floor, astd * 2.2)
            e_a = damp / sig_a
            err += 2.2 * min(70.0, e_a * e_a)
        return err

    def _competition_ok(self, grp_set, cand_indices, my_preds, cand_pts, confirmed, conf_preds, l0_matches):
        ds = []
        for ci, (paz, _, pr) in zip(cand_indices, my_preds):
            pt = cand_pts[ci]
            ds.append((wrap_az(pt["Azimuth"] - paz) / 0.010) ** 2 + ((pt["Distance"] - pr) / 0.008) ** 2)
        my_avg_d = sum(ds) / len(ds)
        if my_avg_d <= 0.25:
            return True
        ref_pt = cand_pts[cand_indices[0]]
        for bo in confirmed:
            if bo in grp_set or bo in l0_matches:
                continue
            oaz, oel, or_ = conf_preds[bo]
            if abs(or_ - ref_pt["Distance"]) > 0.022 or abs(oel - ref_pt["Elevation"]) > 0.085:
                continue
            min_od = 999.0
            for ci in cand_indices:
                pt = cand_pts[ci]
                od = (wrap_az(pt["Azimuth"] - oaz) / 0.010) ** 2 + ((pt["Distance"] - or_) / 0.008) ** 2
                if od < min_od:
                    min_od = od
            if min_od * 1.4 < my_avg_d:
                return False
        return True

    def _level0_atomic_formation_match(self, form_groups, confirmed, cand_pts, cand_scores, scan, occupied):
        l0_matches = {}
        conf_preds = {b: self.tracks[b].predict_sph(scan) for b in confirmed}

        def grp_priority(grp):
            mx = 0
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    lk = self.form_links.get((grp[i], grp[j]))
                    if lk is not None and lk.consistent_obs > mx:
                        mx = lk.consistent_obs
            return -mx

        for grp in sorted(form_groups, key=grp_priority):
            k = len(grp)
            if k == 2:
                b1, b2 = grp[0], grp[1]
                lk = self.form_links.get((b1, b2))
                if lk is None or not self._link_ready(b1, b2, lk, allow_co1=True):
                    continue
                t1, t2 = self.tracks[b1], self.tracks[b2]
                dt_min = min(scan - t1.last_scan, scan - t2.last_scan)
                if dt_min > 4:
                    continue
                p1az, p1el, p1r = conf_preds[b1]
                p2az, p2el, p2r = conf_preds[b2]
                c_az, c_el, c_r = p1az, 0.5 * (p1el + p2el), 0.5 * (p1r + p2r)
                rad = min(1.0 + 0.18 * max(0, dt_min - 1), 1.6) * self.l0_centroid_scale
                g_az = max(self.gate_az, 0.008 / max(c_r, 0.08)) * rad
                g_r = self.gate_r * rad
                g_el = max(self.l0_el_gate, self.gate_el * rad)

                local_c = [
                    ci for ci, p in enumerate(cand_pts)
                    if ci not in occupied and cand_scores[ci] >= self.l0_min_pt_score
                    and abs(p["Distance"] - c_r) <= g_r + 0.012
                    and abs(wrap_az(p["Azimuth"] - c_az)) <= g_az + 0.012
                    and abs(p["Elevation"] - c_el) <= g_el + 0.025
                ]
                if len(local_c) < 2:
                    continue

                pdaz, pdel, pdr, pdamp = lk.predict(scan)
                best_pair = None
                best_score = 26.0
                n_loc = len(local_c)
                for i in range(n_loc):
                    ci1 = local_c[i]
                    pt1 = cand_pts[ci1]
                    for j in range(n_loc):
                        if i == j:
                            continue
                        ci2 = local_c[j]
                        pt2 = cand_pts[ci2]
                        t_err = self._pair_template_err(pt1, pt2, pdaz, pdel, pdr, pdamp, lk.amp_std_ema)
                        if t_err is None:
                            continue
                        d1_az = wrap_az(pt1["Azimuth"] - p1az) / g_az
                        d1_r = (pt1["Distance"] - p1r) / g_r
                        d2_az = wrap_az(pt2["Azimuth"] - p2az) / g_az
                        d2_r = (pt2["Distance"] - p2r) / g_r
                        kin_err = 0.5 * (d1_az * d1_az + d1_r * d1_r + d2_az * d2_az + d2_r * d2_r)
                        if kin_err > self.l0_max_kin_err:
                            continue
                        if abs(0.5 * (pt1["Elevation"] + pt2["Elevation"]) - c_el) > self.l0_el_gate:
                            continue
                        tot = t_err + 4.0 * kin_err + 1.5 * (2.0 - cand_scores[ci1] - cand_scores[ci2])
                        if tot < best_score:
                            best_score = tot
                            best_pair = (ci1, ci2)

                if best_pair is not None:
                    ci1, ci2 = best_pair
                    if self._competition_ok({b1, b2}, (ci1, ci2), (conf_preds[b1], conf_preds[b2]),
                                            cand_pts, confirmed, conf_preds, l0_matches):
                        l0_matches[b1] = ci1
                        l0_matches[b2] = ci2
                        occupied.add(ci1)
                        occupied.add(ci2)
                        lk.consec_partial = 0

            elif k == 3 and self.l0_k3:
                dt_min = min(scan - self.tracks[b].last_scan for b in grp)
                if dt_min > 4:
                    continue
                preds = [conf_preds[b] for b in grp]
                c_az = preds[0][0]
                c_el = sum(p[1] for p in preds) / 3.0
                c_r = sum(p[2] for p in preds) / 3.0
                rad = min(1.0 + 0.18 * max(0, dt_min - 1), 1.6) * self.l0_centroid_scale
                g_az = max(self.gate_az, 0.008 / max(c_r, 0.08)) * rad
                g_r = self.gate_r * rad
                g_el = max(self.l0_el_gate, self.gate_el * rad)

                local_c = [
                    ci for ci, p in enumerate(cand_pts)
                    if ci not in occupied and cand_scores[ci] >= self.l0_min_pt_score
                    and abs(p["Distance"] - c_r) <= g_r + 0.015
                    and abs(wrap_az(p["Azimuth"] - c_az)) <= g_az + 0.015
                    and abs(p["Elevation"] - c_el) <= g_el + 0.030
                ]
                if len(local_c) < 2 or len(local_c) > 10:
                    continue

                matched_tri = False
                if len(local_c) >= 3:
                    ilinks = []
                    for i in range(3):
                        for j in range(i + 1, 3):
                            lk = self.form_links.get((grp[i], grp[j]))
                            if lk is not None and lk.consistent_obs >= self.form_min_consistent:
                                pdaz, pdel, pdr, pdamp = lk.predict(scan)
                                ilinks.append((i, j, pdaz, pdel, pdr, pdamp, lk.amp_std_ema))
                    if len(ilinks) >= 2:
                        best_tri = None
                        best_score = 26.0 * len(ilinks)
                        for ci_tri in itertools.permutations(local_c, 3):
                            t_sum = 0.0
                            ok = True
                            for i, j, pdaz, pdel, pdr, pdamp, astd in ilinks:
                                te = self._pair_template_err(cand_pts[ci_tri[i]], cand_pts[ci_tri[j]],
                                                             pdaz, pdel, pdr, pdamp, astd)
                                if te is None:
                                    ok = False
                                    break
                                t_sum += te
                            if not ok:
                                continue
                            kin_err = sum(
                                ((wrap_az(cand_pts[ci_tri[i]]["Azimuth"] - preds[i][0]) / g_az) ** 2
                                 + ((cand_pts[ci_tri[i]]["Distance"] - preds[i][2]) / g_r) ** 2)
                                for i in range(3)
                            ) / 3.0
                            if kin_err > self.l0_max_kin_err:
                                continue
                            tot = t_sum + 4.0 * kin_err
                            if tot < best_score:
                                best_score = tot
                                best_tri = ci_tri
                        if best_tri is not None:
                            if self._competition_ok(set(grp), best_tri, preds, cand_pts, confirmed, conf_preds, l0_matches):
                                for i in range(3):
                                    l0_matches[grp[i]] = best_tri[i]
                                    occupied.add(best_tri[i])
                                self.group_full_obs[tuple(grp)] = scan
                                matched_tri = True

                # 2-of-3 locked sub-pair template match
                if (not matched_tri) and self.l0_k3_subpair:
                    best_sub = None
                    best_score = 20.0
                    for i in range(3):
                        for j in range(i + 1, 3):
                            b1, b2 = grp[i], grp[j]
                            lk = self.form_links.get((b1, b2))
                            if lk is None or lk.consistent_obs < self.form_min_consistent:
                                continue
                            pdaz, pdel, pdr, pdamp = lk.predict(scan)
                            p1az, _, p1r = preds[i]
                            p2az, _, p2r = preds[j]
                            for ci1 in local_c:
                                pt1 = cand_pts[ci1]
                                for ci2 in local_c:
                                    if ci1 == ci2:
                                        continue
                                    pt2 = cand_pts[ci2]
                                    te = self._pair_template_err(pt1, pt2, pdaz, pdel, pdr, pdamp, lk.amp_std_ema)
                                    if te is None:
                                        continue
                                    kin_err = 0.5 * (
                                        (wrap_az(pt1["Azimuth"] - p1az) / g_az) ** 2 + ((pt1["Distance"] - p1r) / g_r) ** 2
                                        + (wrap_az(pt2["Azimuth"] - p2az) / g_az) ** 2 + ((pt2["Distance"] - p2r) / g_r) ** 2
                                    )
                                    if kin_err > self.l0_max_kin_err:
                                        continue
                                    tot = te + 4.0 * kin_err + 1.5 * (2.0 - cand_scores[ci1] - cand_scores[ci2])
                                    if tot < best_score:
                                        best_score = tot
                                        best_sub = (b1, b2, ci1, ci2, i, j)
                    if best_sub is not None:
                        b1, b2, ci1, ci2, i, j = best_sub
                        if self._competition_ok(set(grp), (ci1, ci2), (preds[i], preds[j]),
                                                cand_pts, confirmed, conf_preds, l0_matches):
                            l0_matches[b1] = ci1
                            l0_matches[b2] = ci2
                            occupied.add(ci1)
                            occupied.add(ci2)

        return l0_matches

    def _solve_bipartite(self, track_ids, cand_pts, cand_scores, scan, gate_scale=1.0,
                         max_cost=1.20, occupied=None, min_pt_score=0.0, is_tentative=False,
                         long_scale=None, ctx_pmap=None, ctx_cset=None):
        if not track_ids or not cand_pts:
            return {}
        if (not is_tentative) and self.mc_add > 0.0:
            max_cost = max_cost + self.mc_add
        aff_w = (self.aff_tent_w if is_tentative else self.aff_bipartite_w) if self._cur_aff else 0.0
        if (not is_tentative) and aff_w > 0.0 and self.aff_mc_add > 0.0:
            max_cost = max_cost + self.aff_mc_add
        use_ctx = (not is_tentative) and (self.ctx_w > 0.0) and bool(ctx_pmap)
        if use_ctx and self.ctx_mc_add > 0.0:
            max_cost = max_cost + self.ctx_mc_add
        occupied = occupied or set()
        ctx_pmap = ctx_pmap or {}
        ctx_cset = ctx_cset or set()
        t_edges = []
        used_cands = {}
        for bid in track_ids:
            t = self.tracks[bid]
            dt = scan - t.last_scan
            paz, pel, pr = t.predict_sph(scan)
            if t.hits == 1:
                eff_scale = max(gate_scale, 1.75)
            elif long_scale is not None and dt >= 3:
                eff_scale = long_scale
            else:
                eff_scale = gate_scale
            rad = min(1.0 + 0.22 * max(0, dt - 1), 2.15) * eff_scale
            rad_sq = rad * rad
            penalty = 0.12 * (1.0 - min(t.strong_hits, 6) / 6.0)
            if (not is_tentative) and t.hits >= 10:
                penalty -= self.age_bonus

            v_xy_sq = t.vx * t.vx + t.vy * t.vy
            use_head = (self.heading_pen_weight > 0.0 and t.hits >= 4 and v_xy_sq > 0.004 * 0.004)
            v_xy = math.sqrt(v_xy_sq) if use_head else 0.0

            edges = []
            for ci, p in enumerate(cand_pts):
                if ci in occupied or cand_scores[ci] < min_pt_score:
                    continue
                if abs(p["Distance"] - pr) > self.gate_r * rad * 1.15:
                    continue
                d_sq = self._norm_dist_sq(paz, pel, pr, p["Azimuth"], p["Elevation"], p["Distance"], pr)
                if d_sq <= rad_sq:
                    if abs(p["Elevation"] - pel) > 0.25:
                        continue
                    cost = (d_sq / rad_sq) + 0.45 * (1.0 - cand_scores[ci]) + penalty
                    if self.r_cost_w > 0.0 and (not is_tentative):
                        dr_norm = (p["Distance"] - pr) / (self.r_sigma * rad)
                        cost += self.r_cost_w * min(1.0, dr_norm * dr_norm)
                    if aff_w > 0.0:
                        p_aff = self._get_pair_aff(bid, ci)
                        cost += aff_w * ((1.0 - p_aff) if p_aff is not None else self.aff_neutral)
                    if use_ctx:
                        p_ctx = ctx_pmap.get((bid, ci))
                        if p_ctx is not None and (self.ctx_all_edges or (bid, ci) in ctx_cset):
                            if self.ctx_mode == "nll":
                                cost += self.ctx_w * min(3.5, -math.log(max(1e-3, p_ctx)))
                            else:
                                cost += self.ctx_w * (1.0 - p_ctx)
                        elif self.ctx_mc_add > 0.0:
                            cost += self.ctx_w * 0.15
                    if use_head:
                        cx, cy, _ = to_cart(p["Azimuth"], t.el, p["Distance"])
                        dx = (cx - t.x) / max(1, dt)
                        dy = (cy - t.y) / max(1, dt)
                        d_xy = math.sqrt(dx * dx + dy * dy)
                        if d_xy > 0.003:
                            cos_h = (t.vx * dx + t.vy * dy) / (v_xy * d_xy)
                            if cos_h < 0.35:
                                cost += self.heading_pen_weight * (0.35 - cos_h)
                    if cost < max_cost:
                        edges.append((cost, ci))
            if edges:
                edges.sort()
                edges = edges[:12]
                t_edges.append((bid, edges))
                for _, ci in edges:
                    if ci not in used_cands:
                        used_cands[ci] = len(used_cands)

        if not t_edges:
            return {}
        n_rows = len(t_edges)
        n_cols = len(used_cands)
        dummy_cost = max_cost
        big = max_cost + 100.0
        mat = np.full((n_rows, n_cols + n_rows), big, dtype=np.float64)
        for r, (_, edges) in enumerate(t_edges):
            mat[r, n_cols + r] = dummy_cost
            for cost, ci in edges:
                mat[r, used_cands[ci]] = cost

        ri, ci_sol = linear_sum_assignment(mat)
        inv_cands = [0] * n_cols
        for orig_ci, col_idx in used_cands.items():
            inv_cands[col_idx] = orig_ci
        matches = {}
        for r, c in zip(ri, ci_sol):
            if c < n_cols and mat[r, c] < dummy_cost:
                matches[t_edges[r][0]] = inv_cands[c]
        return matches

    def _prune_zombie_duplicates(self, form_groups, matches, scan):
        pruned = set()
        for grp in form_groups:
            if len(grp) < 3:
                continue
            gkey = tuple(grp)
            if sum(1 for b in grp if b in matches) == len(grp):
                self.group_full_obs[gkey] = scan
                continue
            last_full = self.group_full_obs.get(gkey)
            if last_full is None:
                self.group_full_obs[gkey] = scan - self.zombie_prune_scans + 2
                continue
            if scan - last_full >= self.zombie_prune_scans:
                candidates_z = sorted(grp, key=lambda b: (1 if b in matches else 0, self.tracks[b].hits, -b))
                bz = candidates_z[0]
                pruned.add(bz)
                if bz in matches:
                    del matches[bz]
        for bz in pruned:
            if bz in self.tracks:
                del self.tracks[bz]
            self.form_links = {k: v for k, v in self.form_links.items() if bz not in k}
        return pruned

    def _optimize_formation_matches(self, form_groups, matches, cand_pts, cand_scores, scan, occupied,
                                    allow_co1=False, allow_guard=True):
        suppressed = set()
        centroid_coasts = []

        for grp in form_groups:
            grp = [b for b in grp if b in self.tracks]
            if len(grp) < 2:
                continue
            k = len(grp)
            matched_b = [b for b in grp if b in matches]
            missed_b = [b for b in grp if b not in matches]

            if allow_guard and self.form_outlier_reject and len(matched_b) == 2:
                b0, b1 = matched_b[0], matched_b[1]
                key = (min(b0, b1), max(b0, b1))
                lk = self.form_links.get(key)
                if (lk is not None and lk.consistent_obs >= self.form_outlier_min_cons
                        and abs(lk.d_az) < 0.010 and abs(lk.d_r) < 0.009
                        and lk.pos_step_ema < 0.0025 and getattr(lk, "consec_split", 0) < 2):
                    pdaz, pdel, pdr, _ = lk.predict(scan)
                    sgn = 1.0 if b0 < b1 else -1.0
                    tdaz, tdel, tdr = sgn * pdaz, sgn * pdel, sgn * pdr
                    sig_az, sig_el, sig_r = self._perm_sigmas(lk)
                    thr_az = max(self.form_outlier_thr_az, sig_az * 4.5)
                    thr_el = max(self.form_outlier_thr_el, sig_el * 5.0)
                    thr_r = max(self.form_outlier_thr_r, sig_r * 4.5)
                    p0, p1 = cand_pts[matches[b0]], cand_pts[matches[b1]]
                    daz01 = wrap_az(p0["Azimuth"] - p1["Azimuth"])
                    del01 = p0["Elevation"] - p1["Elevation"]
                    dr01 = p0["Distance"] - p1["Distance"]
                    ok_a = (abs(wrap_az(daz01 - tdaz)) <= thr_az
                            and abs(del01 - tdel) <= thr_el
                            and abs(dr01 - tdr) <= thr_r)
                    ok_b = (abs(wrap_az(-daz01 - tdaz)) <= thr_az
                            and abs(-del01 - tdel) <= thr_el
                            and abs(-dr01 - tdr) <= thr_r)
                    if ok_a or ok_b:
                        lk.consec_split = 0
                    else:
                        lk.consec_split += 1
                        pred0 = self.tracks[b0].predict_sph(scan)
                        pred1 = self.tracks[b1].predict_sph(scan)
                        errs = []
                        for ck in (matches[b0], matches[b1]):
                            pk = cand_pts[ck]
                            kin_k = min(
                                self._norm_dist_sq(pred0[0], pred0[1], pred0[2], pk["Azimuth"], pk["Elevation"], pk["Distance"], pred0[2]),
                                self._norm_dist_sq(pred1[0], pred1[1], pred1[2], pk["Azimuth"], pk["Elevation"], pk["Distance"], pred1[2]),
                            )
                            aff_k = max(self._get_pair_aff(b0, ck) or 0.5, self._get_pair_aff(b1, ck) or 0.5)
                            errs.append(kin_k + 0.6 * (1.0 - cand_scores[ck]) + 1.0 * (1.0 - aff_k))
                        worst = 0 if errs[0] > errs[1] else 1
                        best = 1 - worst
                        if errs[worst] >= errs[best] + self.form_outlier_margin:
                            b_drop = matched_b[worst]
                            ci_drop = matches.pop(b_drop)
                            occupied.discard(ci_drop)
                            matched_b = [matched_b[best]]
                            missed_b = missed_b + [b_drop]

            if len(matched_b) >= 2:
                m = len(matched_b)
                m_pts = [matches[b] for b in matched_b]
                if m <= 5:
                    links = []
                    for i in range(m):
                        for j in range(i + 1, m):
                            key = (min(matched_b[i], matched_b[j]), max(matched_b[i], matched_b[j]))
                            lk = self.form_links.get(key)
                            if lk is None:
                                continue
                            if self.perm_sep_daz > 0.0:
                                p_i = cand_pts[m_pts[i]]
                                p_j = cand_pts[m_pts[j]]
                                if (abs(wrap_az(p_i["Azimuth"] - p_j["Azimuth"])) > self.perm_sep_daz
                                        or abs(p_i["Distance"] - p_j["Distance"]) > self.perm_sep_dr):
                                    continue
                            if self._link_ready(key[0], key[1], lk, allow_co1=allow_co1):
                                pdaz, pdel, pdr, pdamp = lk.predict(scan)
                                sgn = 1.0 if matched_b[i] < matched_b[j] else -1.0
                                s_az, s_el, s_r = self._perm_sigmas(lk)
                                links.append((i, j, sgn * pdaz, sgn * pdel, sgn * pdr, sgn * pdamp,
                                              lk.amp_std_ema, s_az, s_el, s_r))
                    if links:
                        preds = [self.tracks[b].predict_sph(scan) for b in matched_b]
                        best_perm = tuple(range(m))
                        best_cost = float("inf")
                        for perm in itertools.permutations(range(m)):
                            u_c = 0.0
                            for i in range(m):
                                p = cand_pts[m_pts[perm[i]]]
                                paz, pel, pr = preds[i]
                                u_c += 0.05 * self._norm_dist_sq(paz, pel, pr, p["Azimuth"], p["Elevation"], p["Distance"], pr)
                            p_c = 0.0
                            for i, j, pdaz, pdel, pdr, pdamp, astd, s_az, s_el, s_r in links:
                                p1 = cand_pts[m_pts[perm[i]]]
                                p2 = cand_pts[m_pts[perm[j]]]
                                e_az = (wrap_az(p1["Azimuth"] - p2["Azimuth"]) - pdaz) / s_az
                                e_el = ((p1["Elevation"] - p2["Elevation"]) - pdel) / s_el
                                e_r = ((p1["Distance"] - p2["Distance"]) - pdr) / s_r
                                pc = min(40.0, e_az * e_az) + min(80.0, e_el * e_el) + min(40.0, e_r * e_r)
                                if self.form_use_amp:
                                    sig_a = max(self.amp_sig_floor, astd * 2.5)
                                    e_a = ((p1["Amp"] - p2["Amp"]) - pdamp) / sig_a
                                    pc += 2.5 * min(80.0, e_a * e_a)
                                p_c += pc
                            tot = u_c + p_c
                            if tot < best_cost:
                                best_cost = tot
                                best_perm = perm
                        for i in range(m):
                            matches[matched_b[i]] = m_pts[best_perm[i]]

                if allow_guard and missed_b:
                    for bm in list(missed_b):
                        best_ci = None
                        best_err = 25.0
                        for bh in matched_b:
                            key = (min(bm, bh), max(bm, bh))
                            lk = self.form_links.get(key)
                            if lk is None or lk.consistent_obs < self.form_min_consistent:
                                continue
                            pdaz, pdel, pdr, pdamp = lk.predict(scan)
                            sgn = 1.0 if bm < bh else -1.0
                            ph = cand_pts[matches[bh]]
                            for ci, p in enumerate(cand_pts):
                                if ci in occupied or cand_scores[ci] < 0.05:
                                    continue
                                e = self._pair_template_err(p, ph, sgn * pdaz, sgn * pdel, sgn * pdr, sgn * pdamp, lk.amp_std_ema, lk=lk)
                                if e is not None and e < best_err:
                                    best_err = e
                                    best_ci = ci
                        if best_ci is not None:
                            matches[bm] = best_ci
                            occupied.add(best_ci)
                            matched_b.append(bm)
                            missed_b.remove(bm)

            if allow_guard and len(matched_b) == 1 and k >= 2:
                b_hit = matched_b[0]
                ci = matches[b_hit]
                p = cand_pts[ci]
                best_miss = None
                best_lk = None
                for bm in missed_b:
                    key = (min(b_hit, bm), max(b_hit, bm))
                    lk = self.form_links.get(key)
                    if lk is not None and self._link_ready(key[0], key[1], lk, allow_co1=allow_co1):
                        if best_lk is None or lk.consistent_obs > best_lk.consistent_obs:
                            best_lk = lk
                            best_miss = bm
                if best_lk is not None:
                    b_miss = best_miss
                    lk = best_lk
                    pdaz, pdel, pdr, pdamp = lk.predict(scan)
                    sgn_mh = 1.0 if b_miss < b_hit else -1.0
                    best_ci2 = None
                    best_swap = False
                    best_err = 30.0
                    for ci2, p2 in enumerate(cand_pts):
                        if ci2 in occupied or cand_scores[ci2] < 0.05:
                            continue
                        eA = self._pair_template_err(p2, p, sgn_mh * pdaz, sgn_mh * pdel, sgn_mh * pdr, sgn_mh * pdamp, lk.amp_std_ema, lk=lk)
                        if eA is not None and eA < best_err:
                            best_err, best_ci2, best_swap = eA, ci2, False
                        eB = self._pair_template_err(p, p2, sgn_mh * pdaz, sgn_mh * pdel, sgn_mh * pdr, sgn_mh * pdamp, lk.amp_std_ema, lk=lk)
                        if eB is not None and eB < best_err:
                            best_err, best_ci2, best_swap = eB, ci2, True
                    if best_ci2 is not None:
                        occupied.add(best_ci2)
                        lk.consec_partial = 0
                        if best_swap:
                            matches[b_miss] = ci
                            matches[b_hit] = best_ci2
                        else:
                            matches[b_miss] = best_ci2
                        continue

                    lk.consec_partial += 1
                    if self.form_ambiguity_guard and lk.consec_partial <= self.max_consec_guard:
                        if abs(lk.d_az) < self.form_amb_az and abs(lk.d_r) < self.form_amb_r:
                            th = self.tracks[b_hit]
                            tm = self.tracks[b_miss]
                            haz, _, hr = th.predict_sph(scan)
                            maz, _, mr = tm.predict_sph(scan)
                            eh = (wrap_az(p["Azimuth"] - haz) / 0.0018) ** 2 + ((p["Distance"] - hr) / 0.0006) ** 2
                            em = (wrap_az(p["Azimuth"] - maz) / 0.0018) ** 2 + ((p["Distance"] - mr) / 0.0006) ** 2
                            if em + 1.0 < eh:
                                del matches[b_hit]
                                matches[b_miss] = ci
                                b_hit, b_miss = b_miss, b_hit
                                eh, em = em, eh
                            if (em - eh) < self.form_amb_margin or eh > 6.0:
                                suppressed.add(b_hit)
                                centroid_coasts.append((b_hit, b_miss, ci))

        return suppressed, centroid_coasts

    def _update_seq_memories(self, prev_tracks, points, scores, outputs):
        if self.seq_scorer is None:
            return
        out_by_bid = {o["BatchID"]: o for o in outputs}
        pt_by_pid = {p["PntID"]: (p, s) for p, s in zip(points, scores)}
        for bid, t in self.tracks.items():
            if bid not in prev_tracks:
                p_init = {"Azimuth": t.az, "Elevation": t.raw_el, "Distance": t.r, "Amp": t.amp, "SNR": t.snr}
                self.seq_memories[bid] = TrackSeqMemory(p_init, t.last_score, self.scan)
            elif t.real_last_scan == self.scan:
                o = out_by_bid.get(bid)
                if o is not None and o["PntID"] in pt_by_pid:
                    p_h, sc_h = pt_by_pid[o["PntID"]]
                else:
                    p_h = {"Azimuth": t.az, "Elevation": t.raw_el, "Distance": t.r, "Amp": t.amp, "SNR": t.snr}
                    sc_h = t.last_score
                self.seq_memories[bid].record_hit(t, p_h, sc_h, self.scan)
            else:
                self.seq_memories[bid].record_miss(t, self.scan)
        if self.scan % 25 == 0:
            alive = set(self.tracks.keys()) | set(self.dormant.keys())
            self.seq_memories = {b: m for b, m in self.seq_memories.items() if b in alive}

    def __call__(self, points, scores, pair_aff=None, cur_motion=None):
        self._cur_motion = cur_motion
        prev_tracks = set(self.tracks.keys())
        self.scan += 1
        scan = self.scan
        self._cur_aff = pair_aff or {}
        if not points:
            self._cand_orig_idx = []
            self._update_seq_memories(prev_tracks, (), (), ())
            return []

        cand_pts = []
        cand_scores = []
        cand_orig = []
        for idx_p, (p, s) in enumerate(zip(points, scores)):
            if s >= self.min_cand_score:
                cand_pts.append(p)
                cand_scores.append(s)
                cand_orig.append(idx_p)
        self._cand_orig_idx = cand_orig

        occupied = set()
        confirmed = [b for b, t in self.tracks.items() if self._is_confirmed(t)]
        tentative = [b for b, t in self.tracks.items() if not self._is_confirmed(t)]
        matches = {}
        form_groups = []

        if self.use_formation and len(confirmed) >= 2:
            form_groups = self._active_formation_groups(confirmed, allow_co1=self.parallel_co1_lock)
            if self.use_atomic_l0 and form_groups:
                l0_m = self._level0_atomic_formation_match(form_groups, confirmed, cand_pts, cand_scores, scan, occupied)
                matches.update(l0_m)

        if self.use_formation and self.l0_include_newborn and confirmed and tentative:
            cn_groups = []
            used_cn = set()
            for (b1, b2), lk in self.form_links.items():
                if b1 in used_cn or b2 in used_cn or b1 in matches or b2 in matches:
                    continue
                if b1 not in self.tracks or b2 not in self.tracks:
                    continue
                if scan - lk.last_co_scan > 2:
                    continue
                c1, c2 = self._is_confirmed(self.tracks[b1]), self._is_confirmed(self.tracks[b2])
                if c1 != c2:
                    cn_groups.append([b1, b2])
                    used_cn.add(b1)
                    used_cn.add(b2)
            if cn_groups:
                l0_cn = self._level0_atomic_formation_match(cn_groups, list(self.tracks.keys()),
                                                            cand_pts, cand_scores, scan, occupied)
                matches.update(l0_cn)

        rem_confirmed = [b for b in confirmed if b not in matches]
        ctx_pmap, ctx_cset = self._compute_ctx_scores(
            rem_confirmed, cand_pts, cand_scores, scan, gate_scale=1.0, occupied=occupied
        )
        conf_m = self._solve_bipartite(rem_confirmed, cand_pts, cand_scores, scan,
                                       gate_scale=1.0, max_cost=1.18, occupied=occupied,
                                       ctx_pmap=ctx_pmap, ctx_cset=ctx_cset)
        matches.update(conf_m)
        occupied.update(conf_m.values())

        missed_conf = [b for b in confirmed if b not in matches]
        if missed_conf and self.rescue_scale > 1.0:
            ctx_pmap2, ctx_cset2 = self._compute_ctx_scores(
                missed_conf, cand_pts, cand_scores, scan,
                gate_scale=self.rescue_scale, occupied=occupied,
                min_pt_score=self.rescue_min_score, long_scale=self.rescue_long_scale,
            )
            resc_m = self._solve_bipartite(missed_conf, cand_pts, cand_scores, scan,
                                           gate_scale=self.rescue_scale, max_cost=1.35,
                                           occupied=occupied, min_pt_score=self.rescue_min_score,
                                           long_scale=self.rescue_long_scale,
                                           ctx_pmap=ctx_pmap2, ctx_cset=ctx_cset2)
            matches.update(resc_m)
            occupied.update(resc_m.values())

        if self.filter_low_score_form and form_groups:
            for grp in form_groups:
                for b in grp:
                    ci = matches.get(b)
                    if ci is not None and cand_scores[ci] < self.maintain_score:
                        del matches[b]
                        occupied.discard(ci)

        suppressed_emit = set()
        centroid_coasts = []
        if self.use_formation and form_groups:
            pruned = self._prune_zombie_duplicates(form_groups, matches, scan)
            if pruned:
                confirmed = [b for b in confirmed if b not in pruned]
                form_groups = self._active_formation_groups(confirmed, allow_co1=self.parallel_co1_lock)
            if form_groups:
                suppressed_emit, centroid_coasts = self._optimize_formation_matches(
                    form_groups, matches, cand_pts, cand_scores, scan, occupied,
                    allow_co1=self.parallel_co1_lock,
                )

        # Tier-1 Dormant TSA
        if self.use_tsa and self.dormant:
            self.dormant = {b: t for b, t in self.dormant.items() if scan - t.real_last_scan <= self.max_dormant}
            if self.dormant and cand_pts:
                d_edges = []
                used_c = {}
                for bid, dt_tr in self.dormant.items():
                    elapsed = scan - dt_tr.last_scan
                    rvx, rvy, rvz = dt_tr.robust_vel_cart()
                    paz, pel, pr = from_cart(dt_tr.x + rvx * elapsed, dt_tr.y + rvy * elapsed, dt_tr.z + rvz * elapsed)
                    rad = min(1.0 + 0.08 * elapsed, 2.3)
                    rad_sq = rad * rad
                    v_norm = math.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
                    edges = []
                    for ci, p in enumerate(cand_pts):
                        if ci in occupied or cand_scores[ci] < 0.35:
                            continue
                        if abs(p["Distance"] - pr) > self.gate_r * rad * 1.2:
                            continue
                        d_sq = self._norm_dist_sq(paz, pel, pr, p["Azimuth"], p["Elevation"], p["Distance"], pr)
                        if d_sq <= rad_sq:
                            cx, cy, cz = to_cart(p["Azimuth"], dt_tr.el, p["Distance"])
                            ivx, ivy, ivz = (cx - dt_tr.x) / elapsed, (cy - dt_tr.y) / elapsed, (cz - dt_tr.z) / elapsed
                            iv_norm = math.sqrt(ivx * ivx + ivy * ivy + ivz * ivz)
                            ok = False
                            if v_norm > 0.004 and iv_norm > 1e-4:
                                cos_h = (rvx * ivx + rvy * ivy + rvz * ivz) / (v_norm * iv_norm)
                                spd_ratio = iv_norm / (v_norm + 1e-6)
                                if cos_h >= self.tsa_tier1_cos and 0.45 <= spd_ratio <= 1.8:
                                    ok = True
                            else:
                                dist_3d = math.sqrt((cx - dt_tr.x) ** 2 + (cy - dt_tr.y) ** 2 + (cz - dt_tr.z) ** 2)
                                if dist_3d < 0.04 * (1.0 + 0.1 * elapsed):
                                    ok = True
                            if ok:
                                cost = (d_sq / rad_sq) + 0.45 * (1.0 - cand_scores[ci])
                                if cost < 1.15:
                                    edges.append((cost, ci))
                    if edges:
                        edges.sort()
                        edges = edges[:8]
                        d_edges.append((bid, edges))
                        for _, ci in edges:
                            if ci not in used_c:
                                used_c[ci] = len(used_c)
                if d_edges:
                    nr, nc = len(d_edges), len(used_c)
                    mat = np.full((nr, nc + nr), 100.0, dtype=np.float64)
                    for r, (_, edges) in enumerate(d_edges):
                        mat[r, nc + r] = 1.15
                        for cost, ci in edges:
                            mat[r, used_c[ci]] = cost
                    ri, ci_s = linear_sum_assignment(mat)
                    inv_c = [0] * nc
                    for oci, col in used_c.items():
                        inv_c[col] = oci
                    for r, c in zip(ri, ci_s):
                        if c < nc and mat[r, c] < 1.15:
                            bid = d_edges[r][0]
                            ci = inv_c[c]
                            self.tracks[bid] = self.dormant.pop(bid)
                            matches[bid] = ci
                            occupied.add(ci)

        # Level 4: Tentative Matching + Pre-Lock Permutation
        rem_tent = [b for b in tentative if b not in matches]
        tent_m = self._solve_bipartite(rem_tent, cand_pts, cand_scores, scan,
                                       gate_scale=1.1, max_cost=1.22, occupied=occupied, is_tentative=True)
        matches.update(tent_m)
        occupied.update(tent_m.values())

        if self.use_formation and self.parallel_co1_lock and len(tent_m) >= 2:
            tent_groups = self._active_formation_groups(list(tent_m.keys()), allow_co1=True)
            if tent_groups:
                self._optimize_formation_matches(tent_groups, matches, cand_pts, cand_scores, scan, occupied,
                                                 allow_co1=True, allow_guard=False)

        # Coasting
        coasted_pair_set = set()
        for b1, b2, ci in centroid_coasts:
            coasted_pair_set.add(b1)
            coasted_pair_set.add(b2)
            t1, t2 = self.tracks[b1], self.tracks[b2]
            p = cand_pts[ci]
            lk = self.form_links.get((min(b1, b2), max(b1, b2)))
            if lk is not None:
                pdaz, pdel, pdr, _ = lk.predict(scan)
                ta, tb = (t1, t2) if b1 < b2 else (t2, t1)
                az_a = (p["Azimuth"] + 0.5 * pdaz) % 1.0
                r_a = max(0.001, p["Distance"] + 0.5 * pdr)
                az_b = (p["Azimuth"] - 0.5 * pdaz) % 1.0
                r_b = max(0.001, p["Distance"] - 0.5 * pdr)
                xa, ya, za = to_cart(az_a, ta.el, r_a)
                xb, yb, zb = to_cart(az_b, tb.el, r_b)
                dt_a = max(1, scan - ta.last_scan)
                dt_b = max(1, scan - tb.last_scan)
                ta.vx = 0.75 * ta.vx + 0.25 * (xa - ta.x) / dt_a
                ta.vy = 0.75 * ta.vy + 0.25 * (ya - ta.y) / dt_a
                tb.vx = 0.75 * tb.vx + 0.25 * (xb - tb.x) / dt_b
                tb.vy = 0.75 * tb.vy + 0.25 * (yb - tb.y) / dt_b
                ta.az, ta.r, ta.x, ta.y, ta.z = az_a, r_a, xa, ya, za
                tb.az, tb.r, tb.x, tb.y, tb.z = az_b, r_b, xb, yb, zb
                ta.last_scan = scan
                tb.last_scan = scan

        form_alive_set = set(coasted_pair_set)
        if self.use_formation and form_groups:
            for grp in form_groups:
                hit_members = [b for b in grp if b in matches and b not in suppressed_emit]
                miss_members = [b for b in grp if b not in matches and b not in coasted_pair_set]
                if hit_members and miss_members:
                    form_alive_set.update(miss_members)
                    for bm in miss_members:
                        tm = self.tracks[bm]
                        for bh in hit_members:
                            lk = self.form_links.get((min(bm, bh), max(bm, bh)))
                            if lk is not None and lk.consistent_obs >= self.form_min_consistent:
                                pdaz, pdel, pdr, _ = lk.predict(scan)
                                sgn = 1.0 if bm < bh else -1.0
                                ph = cand_pts[matches[bh]]
                                est_az = (ph["Azimuth"] + sgn * pdaz) % 1.0
                                est_r = max(0.001, ph["Distance"] + sgn * pdr)
                                ex, ey, ez = to_cart(est_az, tm.el, est_r)
                                tm.az, tm.r = est_az, est_r
                                tm.x, tm.y, tm.z = ex, ey, ez
                                tm.vx, tm.vy = self.tracks[bh].vx, self.tracks[bh].vy
                                break

        # Update tracks, Active-Missed + Dormant Tier-2 TSA, Pruning, Outputs
        outputs = []
        matched_tracks_now = []
        reconnected_active_ids = set()

        for bid, t in list(self.tracks.items()):
            if bid in coasted_pair_set or bid in reconnected_active_ids:
                if bid in self.memories:
                    self.memories[bid].record_miss()
                continue
            ci = matches.get(bid)
            real_elapsed = scan - t.real_last_scan
            if ci is None:
                if bid in self.memories:
                    self.memories[bid].record_miss()
                if not self._is_confirmed(t):
                    max_m = self.max_missed_tent if t.max_score >= 0.15 else 1
                    if real_elapsed >= max_m:
                        del self.tracks[bid]
                        self.memories.pop(bid, None)
                        self._eb2_safe.pop(bid, None)
                else:
                    lim = self.form_max_missed if bid in form_alive_set else self.max_missed_conf
                    if real_elapsed > lim:
                        if self.use_tsa and t.hits >= 4:
                            rvx, rvy, rvz = t.robust_vel_cart()
                            t.vx, t.vy, t.vz = rvx, rvy, rvz
                            self.dormant[bid] = t
                        else:
                            self.memories.pop(bid, None)
                        del self.tracks[bid]
                continue

            p = cand_pts[ci]
            sc = cand_scores[ci]
            was_emitted = t.emitted
            t.update(p, sc, scan)
            if bid in self.memories:
                self.memories[bid].record_hit(p, math.hypot(t.vx, t.vy))
            else:
                self.memories[bid] = TrackLongMemory(p)

            if (not was_emitted) and t.hits >= self.confirm_hits and self.use_tsa:
                best_db, best_src, best_llr = None, None, 0.0
                tvx, tvy, tvz = t.robust_vel_cart()
                tv_norm = math.sqrt(tvx * tvx + tvy * tvy + tvz * tvz)
                tsa_pool = [("dormant", db, dt_tr) for db, dt_tr in self.dormant.items()]
                if self.tsa_search_active_missed:
                    for ab, at_tr in self.tracks.items():
                        if (ab != bid and self._is_confirmed(at_tr) and ab not in matches
                                and ab not in coasted_pair_set and ab not in reconnected_active_ids
                                and (scan - at_tr.real_last_scan) >= 2 and at_tr.hits >= 4):
                            tsa_pool.append(("active", ab, at_tr))
                for src, db, dt_tr in tsa_pool:
                    d_elapsed = scan - dt_tr.real_last_scan
                    if d_elapsed > self.max_dormant or d_elapsed <= 0:
                        continue
                    dvx, dvy, dvz = dt_tr.robust_vel_cart()
                    dv_norm = math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
                    mid_t = d_elapsed * 0.5
                    dx_m = (dt_tr.x + dvx * mid_t) - (t.x - tvx * mid_t)
                    dy_m = (dt_tr.y + dvy * mid_t) - (t.y - tvy * mid_t)
                    dz_m = (dt_tr.z + dvz * mid_t) - (t.z - tvz * mid_t)
                    mid_dist = math.sqrt(dx_m * dx_m + dy_m * dy_m + dz_m * dz_m)
                    if mid_dist < self.tsa_tier2_mid_dist:
                        if dv_norm > 0.004 and tv_norm > 0.004:
                            cos_v = (dvx * tvx + dvy * tvy + dvz * tvz) / (dv_norm * tv_norm)
                            if cos_v > self.tsa_tier2_cos:
                                llr = (self.tsa_tier2_mid_dist - mid_dist) / 0.05 + 2.0 * (cos_v - self.tsa_tier2_cos)
                                if llr > best_llr:
                                    best_llr, best_db, best_src = llr, db, src
                        else:
                            direct_d = math.sqrt((dt_tr.x - t.x) ** 2 + (dt_tr.y - t.y) ** 2 + (dt_tr.z - t.z) ** 2)
                            if direct_d < 0.04 * (1.0 + 0.1 * d_elapsed):
                                llr = (0.04 - direct_d) / 0.01 + 1.0
                                if llr > best_llr:
                                    best_llr, best_db, best_src = llr, db, src

                if best_db is not None and best_llr >= self.tsa_llr_th:
                    d_tr = self.dormant.pop(best_db) if best_src == "dormant" else self.tracks.pop(best_db)
                    if best_src == "active":
                        reconnected_active_ids.add(best_db)
                    del self.tracks[bid]
                    self.memories.pop(bid, None)
                    self._eb2_safe.pop(bid, None)
                    d_tr.x, d_tr.y, d_tr.z = t.x, t.y, t.z
                    d_tr.az, d_tr.el, d_tr.raw_el, d_tr.r = t.az, t.el, t.raw_el, t.r
                    d_tr.vx, d_tr.vy, d_tr.vz = t.vx, t.vy, t.vz
                    d_tr.amp, d_tr.snr, d_tr.amp_ema = t.amp, t.snr, t.amp_ema
                    d_tr.last_scan = scan
                    d_tr.real_last_scan = scan
                    d_tr.hits += t.hits
                    d_tr.strong_hits = min(10, d_tr.strong_hits + t.strong_hits)
                    d_tr.emitted = True
                    d_tr.hist_cart = t.hist_cart
                    self.tracks[best_db] = d_tr
                    t = d_tr
                    bid = best_db
                    if bid in self.memories:
                        self.memories[bid].record_hit(p, math.hypot(t.vx, t.vy))

            should_emit = False
            if t.emitted:
                if sc >= self.maintain_score:
                    should_emit = True
            else:
                if t.hits >= self.confirm_hits and (sc >= self.enter_score or t.max_score >= self.enter_score):
                    t.emitted = True
                    if sc >= self.maintain_score:
                        should_emit = True
                elif (self.early_birth_2_score is not None and self._eb2_safe.get(bid, not t.in_protected_birth)
                      and t.hits == 2 and sc >= self.early_birth_2_score):
                    t.emitted = True
                    should_emit = True

            if should_emit:
                outputs.append(_out(p, bid))
            if self._is_confirmed(t) or (self.parallel_co1_lock and t.hits >= 2):
                matched_tracks_now.append(bid)

        # Births
        prot_zones = []
        for b, t in self.tracks.items():
            if self._is_confirmed(t) and (b not in matches or b in coasted_pair_set):
                paz, pel, pr = t.predict_sph(scan)
                dt = scan - t.last_scan
                rad = min(1.0 + 0.22 * max(0, dt - 1), 2.2) * 1.55
                prot_zones.append((paz, pel, pr, rad * rad))
        for b, dt_tr in self.dormant.items():
            elapsed = scan - dt_tr.real_last_scan
            if elapsed <= 15:
                rvx, rvy, rvz = dt_tr.robust_vel_cart()
                paz, pel, pr = from_cart(dt_tr.x + rvx * elapsed, dt_tr.y + rvy * elapsed, dt_tr.z + rvz * elapsed)
                rad = min(1.0 + 0.10 * elapsed, 2.2) * 1.25
                prot_zones.append((paz, pel, pr, rad * rad))

        conf_active_pos = [(t.az, t.el, t.r) for t in self.tracks.values() if self._is_confirmed(t)]
        unmatched = [ci for ci in range(len(cand_pts))
                     if ci not in occupied and cand_scores[ci] >= self.birth_score_min]
        unmatched.sort(key=lambda ci: -cand_scores[ci])
        unmatched = unmatched[:200]

        eb1_mult = self.eb1_safe_mult if self.eb1_safe_mult is not None else self.safe_zone_mult
        eb2_mult = self.eb2_safe_mult if self.eb2_safe_mult is not None else self.safe_zone_mult
        eb1_rad_sq = eb1_mult * eb1_mult
        eb2_rad_sq = eb2_mult * eb2_mult
        born_this_scan = []
        for idx_u, ci in enumerate(unmatched):
            p = cand_pts[ci]
            sc = cand_scores[ci]
            in_prot = False
            for paz, pel, pr, r_sq in prot_zones:
                if abs(p["Distance"] - pr) > self.gate_r * 3.5:
                    continue
                if self._norm_dist_sq(paz, pel, pr, p["Azimuth"], p["Elevation"], p["Distance"], pr) <= r_sq:
                    in_prot = True
                    break
            if in_prot and sc < 0.85:
                continue

            in_safe_eb1 = not in_prot
            in_safe_eb2 = not in_prot
            if not in_prot:
                max_m = max(eb1_mult, eb2_mult)
                for aaz, ael, ar in conf_active_pos:
                    if abs(p["Distance"] - ar) > self.gate_r * max_m:
                        continue
                    d_sq = self._norm_dist_sq(aaz, ael, ar, p["Azimuth"], p["Elevation"], p["Distance"], ar)
                    if in_safe_eb1 and d_sq <= eb1_rad_sq:
                        in_safe_eb1 = False
                    if in_safe_eb2 and eb2_mult > 0.0 and d_sq <= eb2_rad_sq:
                        in_safe_eb2 = False
                    if (not in_safe_eb1) and (not in_safe_eb2):
                        break
            has_sibling = False
            if in_safe_eb1:
                for idx_u2, ci2 in enumerate(unmatched):
                    if idx_u2 == idx_u or cand_scores[ci2] < 0.15:
                        continue
                    p2 = cand_pts[ci2]
                    if (abs(p["Distance"] - p2["Distance"]) < self.sibling_sep_r
                            and abs(wrap_az(p["Azimuth"] - p2["Azimuth"])) < self.sibling_sep_az):
                        has_sibling = True
                        break

            bid = self.next_id
            self.next_id += 1
            t = TrackState(bid, p, sc, scan, in_protected=(not in_safe_eb2),
                           use_imm=self.use_imm, imm_strength=self.imm_strength)
            self.tracks[bid] = t
            self.memories[bid] = TrackLongMemory(p)
            self._eb2_safe[bid] = in_safe_eb2
            born_this_scan.append(bid)

            if (self.early_birth_1_score is not None and in_safe_eb1 and (not has_sibling)
                    and sc >= self.early_birth_1_score):
                t.emitted = True
                outputs.append(_out(p, bid))

        # Update Formation Links
        if self.use_formation:
            if scan % 15 == 0:
                self.form_links = {
                    k: v for k, v in self.form_links.items()
                    if (k[0] in self.tracks and k[1] in self.tracks and scan - v.last_co_scan <= 40)
                }
            if self.parallel_co1_lock and len(born_this_scan) >= 2:
                nb = len(born_this_scan)
                for i in range(nb):
                    b1 = born_this_scan[i]
                    t1 = self.tracks[b1]
                    if t1.last_score < 0.20:
                        continue
                    for j in range(i + 1, nb):
                        b2 = born_this_scan[j]
                        t2 = self.tracks[b2]
                        if t2.last_score < 0.20:
                            continue
                        if (abs(t1.r - t2.r) <= 0.012 and abs(wrap_az(t1.az - t2.az)) <= 0.012
                                and abs(t1.raw_el - t2.raw_el) <= 0.025):
                            key = (b1, b2) if b1 < b2 else (b2, b1)
                            ta, tb = (t1, t2) if b1 < b2 else (t2, t1)
                            self.form_links[key] = FormLink(ta, tb, scan)

            if self.link_conf_to_newborn and born_this_scan and matched_tracks_now:
                for b_new in born_this_scan:
                    tn = self.tracks[b_new]
                    if tn.last_score < 0.25:
                        continue
                    for b_old in matched_tracks_now:
                        to = self.tracks[b_old]
                        if to.last_score < 0.25:
                            continue
                        if (abs(tn.r - to.r) <= 0.012 and abs(wrap_az(tn.az - to.az)) <= 0.012
                                and abs(tn.raw_el - to.raw_el) <= 0.025):
                            key = (b_old, b_new) if b_old < b_new else (b_new, b_old)
                            ta, tb = (to, tn) if b_old < b_new else (tn, to)
                            if key not in self.form_links:
                                self.form_links[key] = FormLink(ta, tb, scan)

            if len(matched_tracks_now) >= 2:
                m_list = matched_tracks_now
                n_m = len(m_list)
                for i in range(n_m):
                    b1 = m_list[i]
                    t1 = self.tracks[b1]
                    if self.filter_low_score_form and t1.last_score < self.maintain_score:
                        continue
                    for j in range(i + 1, n_m):
                        b2 = m_list[j]
                        t2 = self.tracks[b2]
                        if self.filter_low_score_form and t2.last_score < self.maintain_score:
                            continue
                        if abs(t1.r - t2.r) > 0.015:
                            continue
                        if abs(wrap_az(t1.az - t2.az)) > 0.015:
                            continue
                        if abs(t1.raw_el - t2.raw_el) > 0.030:
                            continue
                        key = (b1, b2) if b1 < b2 else (b2, b1)
                        ta, tb = (t1, t2) if b1 < b2 else (t2, t1)
                        if key in self.form_links:
                            self.form_links[key].update(ta, tb, scan)
                        elif ta.last_score >= 0.60 and tb.last_score >= 0.60:
                            self.form_links[key] = FormLink(ta, tb, scan)

        # Record matched/born (CircleID, PntID) observations for neural cross-scan pair affinity
        pt_by_coord = {(p["Azimuth"], p["Distance"], p["Elevation"]): (int(p["CircleID"]), int(p["PntID"]))
                       for p in points}
        for bid, t in self.tracks.items():
            if t.real_last_scan == scan:
                key = pt_by_coord.get((t.az, t.r, t.raw_el))
                if key is not None:
                    dq = self.track_recent_keys.get(bid)
                    if dq is None:
                        dq = deque(maxlen=5)
                        self.track_recent_keys[bid] = dq
                    if not dq or dq[-1] != key:
                        dq.append(key)
        if scan % 20 == 0:
            alive = set(self.tracks.keys()) | set(self.dormant.keys())
            self.track_recent_keys = {b: d for b, d in self.track_recent_keys.items() if b in alive}
            self.memories = {b: m for b, m in self.memories.items() if b in alive}
            self._eb2_safe = {b: v for b, v in self._eb2_safe.items() if b in self.tracks}

        self._update_seq_memories(prev_tracks, points, scores, outputs)
        return outputs


class HouyiPairHead(nn.Module):
    """Lightweight neural cross-scan pair affinity head on top of Houyi (encoded + direct)."""

    def __init__(self, in_dim: int = 1024, emb_dim: int = 64, geom_dim: int = 8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, emb_dim),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(emb_dim * 4 + geom_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.log_scale = nn.Parameter(torch.tensor(2.3))
        self.center = nn.Parameter(torch.tensor(0.5))

    def embed(self, encoded: torch.Tensor, direct: torch.Tensor) -> torch.Tensor:
        h = torch.cat((encoded.float(), direct.float()), dim=-1)
        return F.normalize(self.proj(h), dim=-1)

    def score_pairs(self, z_cur: torch.Tensor, z_hist: torch.Tensor, geom: torch.Tensor):
        cos = (z_cur * z_hist).sum(dim=-1)
        cos_logit = self.log_scale.exp().clamp(max=40.0) * (cos - self.center.clamp(-1.0, 1.0))
        feat = torch.cat((z_cur, z_hist, z_cur * z_hist, (z_cur - z_hist).abs(), geom), dim=-1)
        pair_logit = self.pair_mlp(feat).squeeze(-1)
        return pair_logit, cos_logit


class BackwardMotionHead(nn.Module):
    """CenterTrack-style per-scan backward velocity residual head (v_az*50, v_el*15, v_r*80)."""

    def __init__(self, in_dim: int = 1024, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def forward(self, encoded: torch.Tensor, direct: torch.Tensor) -> torch.Tensor:
        h = torch.cat((encoded.float(), direct.float()), dim=-1)
        return self.net(h)


def _rel_geom_common(f_cur: np.ndarray, f_hist: np.ndarray, max_scans: float, norm_div: float) -> np.ndarray:
    dt = np.clip(f_cur[:, 0] - f_hist[:, 0], 1e-4, max_scans + 3.0)
    d_az = f_cur[:, 4] - f_hist[:, 4]
    d_az = np.where(d_az > 0.5, d_az - 1.0, np.where(d_az < -0.5, d_az + 1.0, d_az))
    d_el = f_cur[:, 5] - f_hist[:, 5]
    d_r = f_cur[:, 3] - f_hist[:, 3]
    d_amp = f_cur[:, 7] - f_hist[:, 7]
    d_snr = f_cur[:, 8] - f_hist[:, 8]
    dt_scan = np.clip(dt * 7.0, 0.5, max_scans)
    return np.column_stack((
        dt_scan / norm_div,
        np.clip(d_az * 50.0, -5.0, 5.0),
        np.clip(d_el * 15.0, -5.0, 5.0),
        np.clip(d_r * 80.0, -5.0, 5.0),
        np.clip(d_amp * 8.0, -5.0, 5.0),
        np.clip(d_snr * 8.0, -5.0, 5.0),
        np.clip((d_az / dt_scan) * 80.0, -5.0, 5.0),
        np.clip((d_r / dt_scan) * 120.0, -5.0, 5.0),
    )).astype(np.float32)


def rel_geom_np(f_cur: np.ndarray, f_hist: np.ndarray) -> np.ndarray:
    """8-dim scale-normalized relative kinematics & signal deltas for 8-scan models."""
    return _rel_geom_common(f_cur, f_hist, max_scans=8.0, norm_div=7.0)


def rel_geom_12s_np(f_cur: np.ndarray, f_hist: np.ndarray) -> np.ndarray:
    """8-dim scale-normalized relative kinematics & signal deltas for 12-scan V18 models."""
    return _rel_geom_common(f_cur, f_hist, max_scans=12.0, norm_div=11.0)


def _fuse_aff(a1: dict, a2: dict) -> dict:
    if not a1:
        return a2
    if not a2:
        return a1
    out = {}
    for oi in set(a1.keys()) | set(a2.keys()):
        d1 = a1.get(oi)
        d2 = a2.get(oi)
        if d1 and d2:
            m = {}
            for k in set(d1.keys()) | set(d2.keys()):
                if k in d1 and k in d2:
                    m[k] = 0.5 * (d1[k] + d2[k])
                else:
                    m[k] = d1.get(k, d2.get(k))
            out[oi] = m
        elif d1:
            out[oi] = d1
        else:
            out[oi] = d2
    return out


class HouyiScorer:
    """Fast bf16 SDPA Houyi v2 point scorer + V18 Joint Pair/Motion Heads + 50%-Overlap Azimuth TTA."""

    def __init__(self, houyi_package: Path, checkpoint_path, device: str = "cuda",
                 weights=None, temperature: float = 1.0, pair_checkpoint=None,
                 use_az_tta: bool = True):
        houyi_package = Path(houyi_package)
        code_dir = str((houyi_package / "code").resolve())
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from model_association.point_cc.feature_augment import FeatureAugmenter
        from model_association.point_cc.model import BertCCModel, BertConfig
        from model_association.radar_definition import PointFeatureIndex
        from model_association.route_point_data_processor import HorizontalCoordPaddedFanSplitter

        if isinstance(checkpoint_path, (str, Path)):
            ckpt_paths = [Path(checkpoint_path)]
        else:
            ckpt_paths = [Path(p) for p in checkpoint_path]
        if weights is None:
            self.weights = [1.0 / len(ckpt_paths)] * len(ckpt_paths)
        else:
            s = sum(weights)
            self.weights = [w / s for w in weights]
        self.temperature = float(temperature)
        self.use_az_tta = bool(use_az_tta)

        model_dir = houyi_package / "Model"
        config = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8-sig"))
        data = json.loads((model_dir / "data_config.json").read_text(encoding="utf-8-sig"))
        indices = tuple(PointFeatureIndex[key] for key in config["input_features"])
        frequencies = {PointFeatureIndex[key]: config["feature_aug_freqs"][key]
                       for key in config["input_aug_features"]}
        self.augmenter = FeatureAugmenter(indices, frequencies, norm_to_01=config["regressor_output_sigmoid"])
        self.extra = config["extra_n_in_features"]
        encoder = BertConfig(
            embedding_size=len(indices),
            hidden_size=config["d_model"],
            num_hidden_layers=config["n_layers"],
            num_attention_heads=config["n_attn_heads"],
            intermediate_size=config["d_ff"],
        )
        encoder._attn_implementation = "sdpa"
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

        def _load_bert(cp_path: Path):
            ckpt = torch.load(cp_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            elif isinstance(ckpt, dict) and "model_state" in ckpt:
                state_dict = ckpt["model_state"]
            else:
                state_dict = ckpt
            out_dim = state_dict["regressor.8.weight"].shape[0]
            model = BertCCModel(
                len(indices) + self.augmenter.n_aug_features + self.extra,
                out_dim,
                encoder,
                config["n_hidden_layers_of_classifier"],
                config["n_hidden_layers_of_regressor"],
                regressor_output_sigmoid=config["regressor_output_sigmoid"],
            )
            model.load_state_dict(state_dict, strict=True)
            last = model.bert_encoder.layer[-1]
            if "forward" in last.__dict__:
                del last.forward
            last.attention.self._forward_hooks.clear()
            last._forward_hooks.clear()
            model.to(self.device).eval()
            return model, ckpt

        self.models = []
        self.model_scans = []
        loaded_ckpts = []
        for cp in ckpt_paths:
            m, ck = _load_bert(cp)
            self.models.append(m)
            sc_n = int(ck.get("scan_count", SCAN_COUNT)) if isinstance(ck, dict) else SCAN_COUNT
            self.model_scans.append(sc_n)
            loaded_ckpts.append((cp.resolve(), ck))

        # Optional cross-scan pair affinity + backward motion heads (V17 pair_head or V18 joint heads)
        self.pair_model = None
        self.pair_head = None
        self.motion_head = None
        self.pair_scan = SCAN_COUNT
        self.pair_use_12s_geom = False
        if pair_checkpoint:
            p_path = Path(pair_checkpoint)
            if p_path.is_file():
                resolved_p = p_path.resolve()
                reused = next((i for i, (rp, _) in enumerate(loaded_ckpts) if rp == resolved_p), None)
                if reused is not None:
                    self.pair_model = self.models[reused]
                    p_ckpt = loaded_ckpts[reused][1]
                else:
                    self.pair_model, p_ckpt = _load_bert(p_path)
                if isinstance(p_ckpt, dict):
                    pair_sd = p_ckpt.get("pair_head_state") or p_ckpt.get("pair_head")
                    if pair_sd is not None:
                        head = HouyiPairHead(in_dim=1024, emb_dim=64, geom_dim=8).to(self.device)
                        head.load_state_dict(pair_sd, strict=True)
                        head.eval()
                        self.pair_head = head
                        self.pair_scan = int(p_ckpt.get("scan_count", SCAN_COUNT))
                        self.pair_use_12s_geom = (self.pair_scan >= 12 or "pair_head_state" in p_ckpt)
                    mot_sd = p_ckpt.get("motion_head_state")
                    if mot_sd is not None:
                        mhead = BackwardMotionHead(in_dim=1024, hidden=256).to(self.device)
                        mhead.load_state_dict(mot_sd, strict=True)
                        mhead.eval()
                        self.motion_head = mhead

        self.max_scan_count = max(self.model_scans + ([self.pair_scan] if self.pair_head is not None else []))

        self.splitter = HorizontalCoordPaddedFanSplitter.generate_fully_covering_splitter(
            r_max=data["max_value"]["r"],
            area_width=np.deg2rad(data["split_azim_width"]),
            pad_width=data["split_pad_width"],
            r1=data["r1"],
        )
        self.corners = self.splitter.circumrect_corners
        self.time_scale = data["normalization_factor"]["time"] * 1000.0
        self.half_az_shift = 0.5 / len(self.splitter.fan_azims)

    def _sectors(self, points, history, return_meta: bool = False, az_shift: float = 0.0):
        if not points:
            return ([], None, None, 0) if return_meta else []
        cloud = [p for scan in (*history, points) for p in scan]
        arr = np.asarray(
            [[p[k] for k in ("Time", "Azimuth", "Elevation", "Distance", "Amp", "SNR")] for p in cloud],
            dtype=np.float64,
        )
        time_col, azimuth, elevation, radius, amp, snr = arr.T
        if az_shift != 0.0:
            azimuth = (azimuth + az_shift) % 1.0
        first_current = len(cloud) - len(points)
        angle = (azimuth - 0.5) * TWO_PI
        deltas = angle[:, None] - self.splitter.fan_azims
        primary = np.argmax(np.cos(deltas), axis=1)
        context = self.splitter.split(radius, angle).astype(bool)
        context[np.arange(len(cloud)), primary] = True
        sectors = []
        for sector in np.unique(primary[first_current:]):
            members = np.flatnonzero(context[:, sector])
            local = (members >= first_current) & (primary[members] == sector)
            xmin, xmax, ymin, ymax = self.corners[sector]
            a = deltas[members, sector]
            r = radius[members]
            features = np.column_stack((
                (time_col[members] - time_col[members].min()) / self.time_scale,
                (r * np.sin(a) - xmin) / (xmax - xmin),
                (r * np.cos(a) - ymin) / (ymax - ymin),
                r, azimuth[members], elevation[members], np.zeros(len(members)),
                amp[members], snr[members], np.full(len(members), 0.5),
            )).astype(np.float32)
            if return_meta:
                sectors.append((features, np.flatnonzero(local), members[local] - first_current, members))
            else:
                sectors.append((features, np.flatnonzero(local), members[local] - first_current))
        if return_meta:
            cids = np.asarray([int(p["CircleID"]) for p in cloud], dtype=np.int32)
            pids = np.asarray([int(p["PntID"]) for p in cloud], dtype=np.int32)
            return sectors, cids, pids, first_current
        return sectors

    def _encode_batch(self, subset):
        count = max(len(s[0]) for s in subset)
        input_data = np.zeros((len(subset), count, 10), dtype=np.float32)
        valid = np.zeros((len(subset), count), dtype=bool)
        for idx, s in enumerate(subset):
            vals = s[0]
            input_data[idx, :len(vals)] = vals
            valid[idx, :len(vals)] = True
        raw = torch.from_numpy(input_data).to(self.device)
        mask = torch.from_numpy(valid).to(self.device)
        inputs = F.pad(torch.cat((raw, self.augmenter.augment(raw)), dim=-1), (0, self.extra))
        return inputs, mask

    @staticmethod
    def _forward_model(model, inputs: torch.Tensor, mask: torch.Tensor):
        hidden = model.input_embedding(inputs)
        attention = (~mask[:, None, None]).to(hidden.dtype) * -10000.0
        encoded = model.bert_encoder(hidden, attention_mask=attention, return_dict=True).last_hidden_state
        direct = model.point_classifier_input_emb(inputs)
        combined = model.point_classifier_concat_emb(torch.cat((encoded, direct), dim=-1))
        logits = model.point_classifier(combined).float()
        return encoded, direct, logits

    def forward_sector_batch(self, subset, model_idx: int = 0):
        """Differentiable forward pass for a mini-batch of sectors (used by training scripts)."""
        inputs, mask = self._encode_batch(subset)
        model = self.models[model_idx]
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            _, _, logits = self._forward_model(model, inputs, mask)
        return logits

    def _eval_point_logits_for_window(self, points, hist_slice, model_indices, az_shift: float = 0.0):
        """Evaluate weighted logit_diff for all models sharing one causal window length."""
        sectors = self._sectors(points, hist_slice, az_shift=az_shift)
        out_ld = np.zeros(len(points), dtype=np.float32)
        bs = 2 if len(hist_slice) >= 9 else SECTOR_BATCH
        for start in range(0, len(sectors), bs):
            subset = sectors[start:start + bs]
            inputs, mask = self._encode_batch(subset)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                ld_sum = None
                for mi in model_indices:
                    _, _, logits = self._forward_model(self.models[mi], inputs, mask)
                    ld = (logits[..., 1] - logits[..., 0]) * self.weights[mi]
                    ld_sum = ld if ld_sum is None else (ld_sum + ld)
            ld_np = ld_sum.cpu().numpy()
            for idx, (_, rows, point_indices) in enumerate(subset):
                out_ld[point_indices] = ld_np[idx, rows]
        return out_ld

    def _eval_pair_window_single(self, points, h_pair, pair_win_models, active_hist_keys,
                                 cand_thresh: float, need_motion: bool, az_shift: float = 0.0):
        sectors, cids, pids, first_current = self._sectors(points, h_pair, return_meta=True, az_shift=az_shift)
        active_mask = np.zeros(first_current, dtype=bool)
        if active_hist_keys and first_current > 0:
            for hi in range(first_current):
                if (int(cids[hi]), int(pids[hi])) in active_hist_keys:
                    active_mask[hi] = True
        need_pairs = bool(np.any(active_mask))

        win_ld = np.zeros(len(points), dtype=np.float32)
        mot_out = np.zeros((len(points), 3), dtype=np.float32) if (need_motion and self.motion_head is not None) else None
        aff_dict = {}

        if self.pair_use_12s_geom:
            geom_fn, dr_win, daz_win, cand_ld_thr = rel_geom_12s_np, 0.036, 0.042, -3.3
        else:
            geom_fn, dr_win, daz_win = rel_geom_np, 0.032, 0.038
            cand_ld_thr = math.log(max(1e-4, cand_thresh) / max(1e-4, 1.0 - cand_thresh)) * self.temperature

        bs = 2 if len(h_pair) >= 9 else SECTOR_BATCH
        for start in range(0, len(sectors), bs):
            subset = sectors[start:start + bs]
            inputs, mask = self._encode_batch(subset)
            pair_enc = pair_dir = pair_ld = ld_sum = None
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                for mi in pair_win_models:
                    model = self.models[mi]
                    enc, dir_e, logits = self._forward_model(model, inputs, mask)
                    raw_ld = logits[..., 1] - logits[..., 0]
                    w = self.weights[mi]
                    ld_sum = (raw_ld * w) if ld_sum is None else (ld_sum + raw_ld * w)
                    if (need_pairs or mot_out is not None) and model is self.pair_model:
                        pair_enc, pair_dir, pair_ld = enc, dir_e, raw_ld
                if (need_pairs or mot_out is not None) and pair_enc is None:
                    pair_enc, pair_dir, p_logits = self._forward_model(self.pair_model, inputs, mask)
                    pair_ld = p_logits[..., 1] - p_logits[..., 0]

            if ld_sum is not None:
                ld_np = ld_sum.cpu().numpy()
                for idx, (_, cur_rows, pt_idx, _) in enumerate(subset):
                    win_ld[pt_idx] = ld_np[idx, cur_rows]

            if mot_out is not None:
                mot_all = self.motion_head(pair_enc, pair_dir).cpu().numpy()
                for idx, (_, cur_rows, pt_idx, _) in enumerate(subset):
                    mot_out[pt_idx] = mot_all[idx, cur_rows]

            if need_pairs:
                z_all = self.pair_head.embed(pair_enc, pair_dir)
                pair_ld_np = pair_ld.cpu().numpy()
                for idx, (feats, cur_rows, pt_idx, members) in enumerate(subset):
                    c_ld = pair_ld_np[idx, cur_rows]
                    act_cur = c_ld >= cand_ld_thr
                    if not np.any(act_cur):
                        continue
                    hist_ok = pair_ld_np[idx, :len(members)] >= cand_ld_thr
                    hist_rows = np.flatnonzero(
                        (members < first_current) & active_mask[np.minimum(members, first_current - 1)] & hist_ok
                    )
                    if not len(hist_rows):
                        continue
                    h_r = feats[hist_rows, 3]
                    h_az = feats[hist_rows, 4]
                    pc_list, ph_list, orig_ci_list = [], [], []
                    for cr, oi in zip(cur_rows[act_cur], pt_idx[act_cur]):
                        dr = np.abs(h_r - feats[cr, 3])
                        daz = np.abs(h_az - feats[cr, 4])
                        daz = np.minimum(daz, 1.0 - daz)
                        near = hist_rows[(dr < dr_win) & (daz < daz_win)]
                        for hr in near:
                            pc_list.append(cr)
                            ph_list.append(hr)
                            orig_ci_list.append(int(oi))
                    if pc_list:
                        pc_a = np.asarray(pc_list, dtype=np.int64)
                        ph_a = np.asarray(ph_list, dtype=np.int64)
                        gm = torch.from_numpy(geom_fn(feats[pc_a], feats[ph_a])).to(self.device)
                        pl, _ = self.pair_head.score_pairs(z_all[idx, pc_a], z_all[idx, ph_a], gm)
                        pp = torch.sigmoid(pl).cpu().numpy()
                        h_mem = members[ph_a]
                        h_cids = cids[h_mem]
                        h_pids = pids[h_mem]
                        for oi, hc, hp, pv in zip(orig_ci_list, h_cids, h_pids, pp):
                            d = aff_dict.get(oi)
                            if d is None:
                                d = {}
                                aff_dict[oi] = d
                            d[(int(hc), int(hp))] = float(pv)

        return win_ld, aff_dict, mot_out

    @torch.inference_mode()
    def scores(self, points, history):
        if not points:
            return []
        hist_list = list(history)
        by_sc = defaultdict(list)
        for mi, sc_n in enumerate(self.model_scans):
            by_sc[sc_n].append(mi)
        shifts = (0.0, self.half_az_shift) if self.use_az_tta else (0.0,)
        total_ld = np.zeros(len(points), dtype=np.float32)
        for sc_n, m_idxs in by_sc.items():
            h_sub = hist_list[-(sc_n - 1):] if sc_n > 1 else []
            ld_acc = None
            for sh in shifts:
                ld_s = self._eval_point_logits_for_window(points, h_sub, m_idxs, az_shift=sh)
                ld_acc = ld_s if ld_acc is None else (ld_acc + ld_s)
            total_ld += ld_acc / len(shifts)
        if self.temperature != 1.0:
            total_ld = total_ld / self.temperature
        probs = 1.0 / (1.0 + np.exp(-np.clip(total_ld, -30.0, 30.0)))
        return probs.tolist()

    @torch.inference_mode()
    def score_and_affinity(self, points, history, active_hist_keys=None, cand_thresh: float = 0.035,
                           return_motion: bool = False):
        """Return (probs_list, pair_aff_dict[, cur_motion]) with 50%-overlap Azimuth TTA."""
        if not points:
            return ([], {}, None) if return_motion else ([], {})
        if self.pair_head is None or self.pair_model is None or not history:
            probs = self.scores(points, history)
            return (probs, {}, None) if return_motion else (probs, {})
        hist_list = list(history)
        by_sc = defaultdict(list)
        for mi, sc_n in enumerate(self.model_scans):
            by_sc[sc_n].append(mi)
        shifts = (0.0, self.half_az_shift) if self.use_az_tta else (0.0,)

        total_ld = np.zeros(len(points), dtype=np.float32)
        for sc_n, m_idxs in by_sc.items():
            if sc_n != self.pair_scan:
                h_sub = hist_list[-(sc_n - 1):] if sc_n > 1 else []
                ld_acc = None
                for sh in shifts:
                    ld_s = self._eval_point_logits_for_window(points, h_sub, m_idxs, az_shift=sh)
                    ld_acc = ld_s if ld_acc is None else (ld_acc + ld_s)
                total_ld += ld_acc / len(shifts)

        h_pair = hist_list[-(self.pair_scan - 1):] if self.pair_scan > 1 else []
        pair_win_models = by_sc.get(self.pair_scan, [])
        need_motion = bool(return_motion and self.motion_head is not None)
        ld_p_acc = None
        aff_acc = None
        mot_acc = None
        for sh in shifts:
            w_ld, w_aff, w_mot = self._eval_pair_window_single(
                points, h_pair, pair_win_models, active_hist_keys,
                cand_thresh=cand_thresh, need_motion=need_motion, az_shift=sh,
            )
            ld_p_acc = w_ld if ld_p_acc is None else (ld_p_acc + w_ld)
            aff_acc = w_aff if aff_acc is None else _fuse_aff(aff_acc, w_aff)
            if w_mot is not None:
                mot_acc = w_mot if mot_acc is None else (mot_acc + w_mot)

        total_ld += ld_p_acc / len(shifts)
        cur_motion = (mot_acc / len(shifts)) if mot_acc is not None else None
        if self.temperature != 1.0:
            total_ld = total_ld / self.temperature
        out = 1.0 / (1.0 + np.exp(-np.clip(total_ld, -30.0, 30.0)))
        if return_motion:
            return out.tolist(), (aff_acc or {}), cur_motion
        return out.tolist(), (aff_acc or {})


DEFAULT_TEMPERATURE = 1.15
DEFAULT_ENSEMBLE_TEMPERATURE = 0.90
DEFAULT_V18_ENSEMBLE_WEIGHTS = (0.30, 0.70)


def resolve_conflict_checkpoint():
    """Locate calibrated ContextualConflictNet V17 checkpoint if available."""
    c_env = os.environ.get("B02_CONFLICT_CHECKPOINT")
    if c_env is not None:
        return Path(c_env) if c_env.strip() else None
    starter_root = Path(__file__).resolve().parents[1]
    workspace = Path(__file__).resolve().parents[3]
    cands = [
        starter_root / "assets" / "contextual_conflict_v17.pt",
        workspace / "output" / "houyi_pair_170" / "contextual_conflict_v17.pt",
    ]
    return next((p for p in cands if p.is_file()), None)


def resolve_seq_checkpoint():
    """Locate V18-C TrackSeqCrossAttnNet checkpoint if available."""
    s_env = os.environ.get("B02_SEQ_CHECKPOINT")
    if s_env is not None:
        return Path(s_env) if s_env.strip() else None
    starter_root = Path(__file__).resolve().parents[1]
    workspace = Path(__file__).resolve().parents[3]
    cands = [
        starter_root / "assets" / "track_seq_cross_attn_v18.pt",
        workspace / "output" / "houyi_v18_170" / "track_seq_cross_attn_v18.pt",
    ]
    return next((p for p in cands if p.is_file()), None)


def build_circle_handler(houyi_package: Path, checkpoint_path, device: str = "cuda",
                         weights=None, temperature: float = DEFAULT_TEMPERATURE,
                         pair_checkpoint=None, conflict_checkpoint=None,
                         seq_checkpoint=None, use_az_tta: bool = True, **tracker_overrides):
    """Build a stateful (points, state) -> list[dict] handler compatible with b02 replay and official TCP client."""
    scorer = HouyiScorer(houyi_package, checkpoint_path, device=device,
                         weights=weights, temperature=temperature,
                         pair_checkpoint=pair_checkpoint, use_az_tta=use_az_tta)
    seq_scorer = (
        FastSeqCrossAttnScorer(seq_checkpoint, device=device)
        if seq_checkpoint and Path(seq_checkpoint).is_file() else None
    )
    conflict_net = (
        FastNumpyConflictNet(conflict_checkpoint)
        if (seq_scorer is None and conflict_checkpoint and Path(conflict_checkpoint).is_file()) else None
    )
    eff_overrides = dict(tracker_overrides)
    if seq_scorer is None and conflict_net is not None:
        eff_overrides.setdefault("ctx_w", 0.35)
        eff_overrides.setdefault("ctx_mc_add", 0.25)

    def handle(points, state):
        if not state:
            state["history"] = deque(maxlen=max(1, scorer.max_scan_count - 1))
            state["tracker"] = MasterTracker(conflict_net=conflict_net, seq_scorer=seq_scorer, **eff_overrides)
        history = state["history"]
        tracker = state["tracker"]
        if scorer.pair_head is not None:
            probs, pair_aff, cur_motion = scorer.score_and_affinity(
                points, history, active_hist_keys=tracker.active_hist_keys(),
                cand_thresh=max(0.035, tracker.min_cand_score), return_motion=True,
            )
            history.append(points)
            return tracker(points, probs, pair_aff=pair_aff, cur_motion=cur_motion)
        probs = scorer.scores(points, history)
        history.append(points)
        return tracker(points, probs)

    return handle


def resolve_default_paths():
    """Locate HouyiBypassV2 package, point checkpoint(s) (V18+v3 or V17 fallback), and pair checkpoint."""
    starter_root = Path(__file__).resolve().parents[1]
    workspace = Path(__file__).resolve().parents[3]
    houyi_env = os.environ.get("B02_HOUYI_DIR")
    if houyi_env:
        houyi_pkg = Path(houyi_env)
    else:
        h_cands = [
            starter_root / "assets" / "HouyiBypassV2",
            workspace / "智能体应用赛题B02--目标检测跟踪--后羿基础模型" / "HouyiV2-publish" / "HouyiBypassV2",
        ]
        houyi_pkg = next((p for p in h_cands if p.is_dir()), h_cands[-1])

    p_v18 = next((p for p in (
        starter_root / "assets" / "houyi-v18-joint-12scan.pt",
        workspace / "output" / "houyi_v18_170" / "houyi-v18-joint-12scan.pt",
    ) if p.is_file()), None)

    ckpt_env = os.environ.get("B02_CHECKPOINT")
    if ckpt_env:
        parts = [Path(s.strip()) for s in ckpt_env.split(",") if s.strip()]
        ckpt_path = parts[0] if len(parts) == 1 else parts
    else:
        c8 = [
            starter_root / "assets" / "houyi-dense-v3-e02.pt",
            workspace / "output" / "houyi_dense_170" / "houyi-dense-v3-e02.pt",
            workspace / "output" / "houyi_dense_170" / "houyi-dense-e02.pt",
        ]
        p8 = next((p for p in c8 if p.is_file()), c8[0])
        if p8.is_file() and p_v18 is not None:
            ckpt_path = [p8, p_v18]
        else:
            c12 = [
                starter_root / "assets" / "houyi-v4-12scan-e02.pt",
                workspace / "output" / "houyi_v4_12scan_170" / "houyi-v4-12scan-e02.pt",
            ]
            p12 = next((p for p in c12 if p.is_file()), None)
            ckpt_path = [p8, p12] if (p8.is_file() and p12 is not None) else p8

    pair_env = os.environ.get("B02_PAIR_CHECKPOINT")
    if pair_env is not None:
        pair_path = Path(pair_env) if pair_env.strip() else None
    elif p_v18 is not None:
        pair_path = p_v18
    else:
        p_cands = [
            starter_root / "assets" / "houyi-pair-e02.pt",
            workspace / "output" / "houyi_pair_170" / "houyi-pair-e02.pt",
        ]
        pair_path = next((p for p in p_cands if p.is_file()), None)
    return houyi_pkg, ckpt_path, pair_path


def load_tracker(checkpoint=None, device: str = "auto", temperature=None,
                 houyi_dir=None, pair_checkpoint="auto", conflict_checkpoint="auto",
                 seq_checkpoint="auto", weights=None, use_az_tta: bool = True, **tracker_overrides):
    """CLI/submission loader returning (handler, config_dict)."""
    default_houyi, default_ckpt, default_pair = resolve_default_paths()
    houyi_pkg = Path(houyi_dir) if houyi_dir else default_houyi
    if checkpoint is None:
        ckpt_path = default_ckpt
    elif isinstance(checkpoint, (list, tuple)):
        ckpt_path = [Path(p) for p in checkpoint]
    else:
        s = str(checkpoint)
        parts = [Path(x.strip()) for x in s.split(",") if x.strip()]
        ckpt_path = parts[0] if len(parts) == 1 else parts
    pair_path = default_pair if pair_checkpoint == "auto" else (Path(pair_checkpoint) if pair_checkpoint else None)
    conflict_path = resolve_conflict_checkpoint() if conflict_checkpoint == "auto" else (Path(conflict_checkpoint) if conflict_checkpoint else None)
    seq_path = resolve_seq_checkpoint() if seq_checkpoint == "auto" else (Path(seq_checkpoint) if seq_checkpoint else None)

    is_ens = isinstance(ckpt_path, (list, tuple)) and len(ckpt_path) > 1
    if weights is None:
        w_env = os.environ.get("B02_WEIGHTS")
        if w_env:
            eff_weights = [float(x) for x in w_env.split(",") if x.strip()]
        elif is_ens and len(ckpt_path) == 2 and "v18" in ckpt_path[1].name.lower():
            eff_weights = list(DEFAULT_V18_ENSEMBLE_WEIGHTS)
        else:
            eff_weights = None
    else:
        eff_weights = list(weights)

    if temperature is None:
        t_env = os.environ.get("B02_TEMPERATURE")
        temp = float(t_env) if t_env else (DEFAULT_ENSEMBLE_TEMPERATURE if is_ens else DEFAULT_TEMPERATURE)
    else:
        temp = float(temperature)

    tta_env = os.environ.get("B02_AZ_TTA")
    eff_tta = use_az_tta if tta_env is None else (tta_env.strip().lower() not in ("0", "false", "no", ""))

    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device
    handler = build_circle_handler(houyi_pkg, ckpt_path, device=dev,
                                   weights=eff_weights, temperature=temp, pair_checkpoint=pair_path,
                                   conflict_checkpoint=conflict_path, seq_checkpoint=seq_path,
                                   use_az_tta=eff_tta, **tracker_overrides)
    if is_ens and seq_path and pair_path and "v18" in pair_path.name.lower():
        t_name = "master-v19.5-v18joint-tta-seqattn-outlier"
    elif is_ens and pair_path and conflict_path:
        t_name = "master-v17-multiwindow-imm-pair-conflict"
    else:
        t_name = "master-legacy"
    cfg = {
        "tracker": t_name,
        "checkpoint": [str(p) for p in ckpt_path] if is_ens else str(ckpt_path),
        "weights": eff_weights,
        "pair_checkpoint": str(pair_path) if pair_path else None,
        "seq_checkpoint": str(seq_path) if seq_path else None,
        "conflict_checkpoint": str(conflict_path) if (not seq_path and conflict_path) else None,
        "houyi_package": str(houyi_pkg),
        "device": dev,
        "temperature": temp,
        "use_az_tta": eff_tta,
        "params": {**DEFAULT_TRACKER_PARAMS, **tracker_overrides},
    }
    return handler, cfg
    return handler, cfg
