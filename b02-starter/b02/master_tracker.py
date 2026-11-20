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
    "early_birth_2_score": 0.55,
    "safe_zone_mult": 2.8,
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
    )

    def __init__(self, batch, p, score, scan, in_protected=False):
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

    def predict_sph(self, scan: int):
        dt = scan - self.last_scan
        if dt <= 0 or self.hits < 2:
            return self.az, self.el, self.r
        return from_cart(self.x + self.vx * dt, self.y + self.vy * dt, self.z + self.vz * dt)

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
        self.raw_el = raw_el
        nx, ny, nz = to_cart(naz, raw_el, nr)
        g = 1.0 if self.hits == 1 else (0.6 if self.hits == 2 else vel_gain)
        trust = min(1.0, max(0.15, score / 0.50))
        eff_g = g * trust
        self.vx = (1.0 - eff_g) * self.vx + eff_g * ((nx - self.x) / dt)
        self.vy = (1.0 - eff_g) * self.vy + eff_g * ((ny - self.y) / dt)
        self.vz = (1.0 - eff_g) * self.vz + eff_g * ((nz - self.z) / dt)
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
        step_err = math.sqrt((new_daz - pdaz) ** 2 + (new_del - pdel) ** 2 + (new_dr - pdr) ** 2) / dt
        amp_err = abs(new_damp - pdamp)

        if self.consistent_obs >= 2:
            if step_err > 0.0045 or (self.amp_std_ema < 0.015 and amp_err > 0.035):
                return

        if step_err < 0.0030 and amp_err < 0.04:
            self.consistent_obs += 1
        else:
            self.consistent_obs = max(0, self.consistent_obs - 1)

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


class MasterTracker:
    def __init__(self, **overrides):
        params = {**DEFAULT_TRACKER_PARAMS, **overrides}
        for k, v in params.items():
            setattr(self, k, v)
        self.scan = 0
        self.next_id = 1
        self.tracks: dict[int, TrackState] = {}
        self.dormant: dict[int, TrackState] = {}
        self.form_links: dict[tuple[int, int], FormLink] = {}
        self.group_full_obs: dict[tuple[int, ...], int] = {}

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

    def _pair_template_err(self, p1, p2, pdaz, pdel, pdr, pdamp, astd):
        da = abs(wrap_az((p1["Azimuth"] - p2["Azimuth"]) - pdaz))
        de = abs((p1["Elevation"] - p2["Elevation"]) - pdel)
        dr = abs((p1["Distance"] - p2["Distance"]) - pdr)
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
                         long_scale=None):
        if not track_ids or not cand_pts:
            return {}
        if (not is_tentative) and self.mc_add > 0.0:
            max_cost = max_cost + self.mc_add
        occupied = occupied or set()
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
                                    allow_co1=False):
        suppressed = set()
        centroid_coasts = []

        for grp in form_groups:
            grp = [b for b in grp if b in self.tracks]
            if len(grp) < 2:
                continue
            k = len(grp)
            matched_b = [b for b in grp if b in matches]
            missed_b = [b for b in grp if b not in matches]

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
                            if self._link_ready(key[0], key[1], lk, allow_co1=allow_co1):
                                pdaz, pdel, pdr, pdamp = lk.predict(scan)
                                sgn = 1.0 if matched_b[i] < matched_b[j] else -1.0
                                links.append((i, j, sgn * pdaz, sgn * pdel, sgn * pdr, sgn * pdamp, lk.amp_std_ema))
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
                            for i, j, pdaz, pdel, pdr, pdamp, astd in links:
                                p1 = cand_pts[m_pts[perm[i]]]
                                p2 = cand_pts[m_pts[perm[j]]]
                                e_az = (wrap_az(p1["Azimuth"] - p2["Azimuth"]) - pdaz) / 0.0015
                                e_el = ((p1["Elevation"] - p2["Elevation"]) - pdel) / 0.0005
                                e_r = ((p1["Distance"] - p2["Distance"]) - pdr) / 0.0012
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

                if missed_b:
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
                                e = self._pair_template_err(p, ph, sgn * pdaz, sgn * pdel, sgn * pdr, sgn * pdamp, lk.amp_std_ema)
                                if e is not None and e < best_err:
                                    best_err = e
                                    best_ci = ci
                        if best_ci is not None:
                            matches[bm] = best_ci
                            occupied.add(best_ci)
                            matched_b.append(bm)
                            missed_b.remove(bm)

            elif len(matched_b) == 1 and k >= 2:
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
                        eA = self._pair_template_err(p2, p, sgn_mh * pdaz, sgn_mh * pdel, sgn_mh * pdr, sgn_mh * pdamp, lk.amp_std_ema)
                        if eA is not None and eA < best_err:
                            best_err, best_ci2, best_swap = eA, ci2, False
                        eB = self._pair_template_err(p, p2, sgn_mh * pdaz, sgn_mh * pdel, sgn_mh * pdr, sgn_mh * pdamp, lk.amp_std_ema)
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

    def __call__(self, points, scores):
        self.scan += 1
        scan = self.scan
        if not points:
            return []

        cand_pts = []
        cand_scores = []
        for p, s in zip(points, scores):
            if s >= self.min_cand_score:
                cand_pts.append(p)
                cand_scores.append(s)

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
        conf_m = self._solve_bipartite(rem_confirmed, cand_pts, cand_scores, scan,
                                       gate_scale=1.0, max_cost=1.18, occupied=occupied)
        matches.update(conf_m)
        occupied.update(conf_m.values())

        missed_conf = [b for b in confirmed if b not in matches]
        if missed_conf and self.rescue_scale > 1.0:
            resc_m = self._solve_bipartite(missed_conf, cand_pts, cand_scores, scan,
                                           gate_scale=self.rescue_scale, max_cost=1.35,
                                           occupied=occupied, min_pt_score=self.rescue_min_score,
                                           long_scale=self.rescue_long_scale)
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
                                                 allow_co1=True)

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
                continue
            ci = matches.get(bid)
            real_elapsed = scan - t.real_last_scan
            if ci is None:
                if not self._is_confirmed(t):
                    max_m = self.max_missed_tent if t.max_score >= 0.15 else 1
                    if real_elapsed >= max_m:
                        del self.tracks[bid]
                else:
                    lim = self.form_max_missed if bid in form_alive_set else self.max_missed_conf
                    if real_elapsed > lim:
                        if self.use_tsa and t.hits >= 4:
                            rvx, rvy, rvz = t.robust_vel_cart()
                            t.vx, t.vy, t.vz = rvx, rvy, rvz
                            self.dormant[bid] = t
                        del self.tracks[bid]
                continue

            p = cand_pts[ci]
            sc = cand_scores[ci]
            was_emitted = t.emitted
            t.update(p, sc, scan)

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

            should_emit = False
            if t.emitted:
                if sc >= self.maintain_score:
                    should_emit = True
            else:
                if t.hits >= self.confirm_hits and (sc >= self.enter_score or t.max_score >= self.enter_score):
                    t.emitted = True
                    if sc >= self.maintain_score:
                        should_emit = True
                elif (self.early_birth_2_score is not None and (not t.in_protected_birth)
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

        safe_rad_sq = self.safe_zone_mult * self.safe_zone_mult
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

            in_safe = not in_prot
            if in_safe:
                for aaz, ael, ar in conf_active_pos:
                    if abs(p["Distance"] - ar) > self.gate_r * self.safe_zone_mult:
                        continue
                    if self._norm_dist_sq(aaz, ael, ar, p["Azimuth"], p["Elevation"], p["Distance"], ar) <= safe_rad_sq:
                        in_safe = False
                        break
            has_sibling = False
            if in_safe:
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
            t = TrackState(bid, p, sc, scan, in_protected=(not in_safe))
            self.tracks[bid] = t
            born_this_scan.append(bid)

            if (self.early_birth_1_score is not None and in_safe and (not has_sibling)
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

        return outputs


class HouyiScorer:
    """Fast bf16 SDPA Houyi v2 point scorer (single model or logit-averaged ensemble)."""

    def __init__(self, houyi_package: Path, checkpoint_path, device: str = "cuda",
                 weights=None, temperature: float = 1.0):
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
        self.models = []
        for cp in ckpt_paths:
            ckpt = torch.load(cp, map_location="cpu", weights_only=True)
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
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
            self.models.append(model)

        self.splitter = HorizontalCoordPaddedFanSplitter.generate_fully_covering_splitter(
            r_max=data["max_value"]["r"],
            area_width=np.deg2rad(data["split_azim_width"]),
            pad_width=data["split_pad_width"],
            r1=data["r1"],
        )
        self.corners = self.splitter.circumrect_corners
        self.time_scale = data["normalization_factor"]["time"] * 1000.0

    def _sectors(self, points, history):
        if not points:
            return []
        cloud = np.asarray(
            [[p[k] for k in ("Time", "Azimuth", "Elevation", "Distance", "Amp", "SNR")]
             for scan in (*history, points) for p in scan],
            dtype=np.float64,
        )
        time_col, azimuth, elevation, radius, amp, snr = cloud.T
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
            sectors.append((features, np.flatnonzero(local), members[local] - first_current))
        return sectors

    def forward_sector_batch(self, subset, model_idx: int = 0):
        """Differentiable forward pass for a mini-batch of sectors (used by training & evaluation)."""
        count = max(len(v) for v, _, _ in subset)
        input_data = np.zeros((len(subset), count, 10), dtype=np.float32)
        valid = np.zeros((len(subset), count), dtype=bool)
        for idx, (vals, _, _) in enumerate(subset):
            input_data[idx, :len(vals)] = vals
            valid[idx, :len(vals)] = True
        raw = torch.from_numpy(input_data).to(self.device)
        mask = torch.from_numpy(valid).to(self.device)
        inputs = F.pad(torch.cat((raw, self.augmenter.augment(raw)), dim=-1), (0, self.extra))
        model = self.models[model_idx]
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            hidden = model.input_embedding(inputs)
            attention = (~mask[:, None, None]).to(hidden.dtype) * -10000.0
            encoded = model.bert_encoder(hidden, attention_mask=attention, return_dict=True).last_hidden_state
            direct = model.point_classifier_input_emb(inputs)
            combined = model.point_classifier_concat_emb(torch.cat((encoded, direct), dim=-1))
            logits = model.point_classifier(combined)
        return logits.float()

    @torch.inference_mode()
    def scores(self, points, history):
        if not points:
            return []
        sectors = self._sectors(points, history)
        out = np.empty(len(points), dtype=np.float32)
        for start in range(0, len(sectors), SECTOR_BATCH):
            subset = sectors[start:start + SECTOR_BATCH]
            count = max(len(v) for v, _, _ in subset)
            input_data = np.zeros((len(subset), count, 10), dtype=np.float32)
            valid = np.zeros((len(subset), count), dtype=bool)
            for idx, (vals, _, _) in enumerate(subset):
                input_data[idx, :len(vals)] = vals
                valid[idx, :len(vals)] = True
            raw = torch.from_numpy(input_data).to(self.device)
            mask = torch.from_numpy(valid).to(self.device)
            inputs = F.pad(torch.cat((raw, self.augmenter.augment(raw)), dim=-1), (0, self.extra))
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                logit_diff_sum = None
                for model, w in zip(self.models, self.weights):
                    hidden = model.input_embedding(inputs)
                    attention = (~mask[:, None, None]).to(hidden.dtype) * -10000.0
                    encoded = model.bert_encoder(hidden, attention_mask=attention, return_dict=True).last_hidden_state
                    direct = model.point_classifier_input_emb(inputs)
                    combined = model.point_classifier_concat_emb(torch.cat((encoded, direct), dim=-1))
                    logits = model.point_classifier(combined).float()
                    ld = (logits[..., 1] - logits[..., 0]) * w
                    logit_diff_sum = ld if logit_diff_sum is None else (logit_diff_sum + ld)
            if self.temperature != 1.0:
                logit_diff_sum = logit_diff_sum / self.temperature
            probs = torch.sigmoid(logit_diff_sum).cpu().numpy()
            for idx, (_, rows, point_indices) in enumerate(subset):
                out[point_indices] = probs[idx, rows]
        return out.tolist()


DEFAULT_TEMPERATURE = 1.15


def build_circle_handler(houyi_package: Path, checkpoint_path, device: str = "cuda",
                         weights=None, temperature: float = DEFAULT_TEMPERATURE, **tracker_overrides):
    """Build a stateful (points, state) -> list[dict] handler compatible with b02 replay and official TCP client."""
    scorer = HouyiScorer(houyi_package, checkpoint_path, device=device,
                         weights=weights, temperature=temperature)

    def handle(points, state):
        if not state:
            state["history"] = deque(maxlen=SCAN_COUNT - 1)
            state["tracker"] = MasterTracker(**tracker_overrides)
        history = state["history"]
        tracker = state["tracker"]
        probs = scorer.scores(points, history)
        history.append(points)
        return tracker(points, probs)

    return handle


def resolve_default_paths():
    """Locate HouyiBypassV2 package and default Houyi-Dense-v3 checkpoint via env vars or workspace layout."""
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
    ckpt_env = os.environ.get("B02_CHECKPOINT")
    if ckpt_env:
        ckpt_path = Path(ckpt_env)
    else:
        candidates = [
            starter_root / "assets" / "houyi-dense-v3-e02.pt",
            workspace / "output" / "houyi_dense_170" / "houyi-dense-v3-e02.pt",
            workspace / "output" / "houyi_dense_170" / "houyi-dense-e02.pt",
        ]
        ckpt_path = next((p for p in candidates if p.is_file()), candidates[0])
    return houyi_pkg, ckpt_path


def load_tracker(checkpoint=None, device: str = "auto", temperature: float = DEFAULT_TEMPERATURE,
                 houyi_dir=None, **tracker_overrides):
    """CLI/submission loader returning (handler, config_dict)."""
    default_houyi, default_ckpt = resolve_default_paths()
    houyi_pkg = Path(houyi_dir) if houyi_dir else default_houyi
    ckpt_path = Path(checkpoint) if checkpoint else default_ckpt
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device
    handler = build_circle_handler(houyi_pkg, ckpt_path, device=dev,
                                   temperature=temperature, **tracker_overrides)
    cfg = {
        "tracker": "master-v14",
        "checkpoint": str(ckpt_path),
        "houyi_package": str(houyi_pkg),
        "device": dev,
        "temperature": temperature,
        "params": {**DEFAULT_TRACKER_PARAMS, **tracker_overrides},
    }
    return handler, cfg
