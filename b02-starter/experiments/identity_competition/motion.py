"""Fit a past-only straight-line reference to trusted memory observations."""
import numpy as np


def memory_motion(memory, when):
    entries = list(memory.entries)
    if len(entries) < 4:
        return None
    xyz = np.stack([e[0] for e in entries]).astype(float)
    time = np.asarray([e[2]-when for e in entries], dtype=float)
    centered = time-time.mean()
    velocity = (centered[:, None]*(xyz-xyz.mean(0))).sum(0)/max(float((centered**2).sum()), 1e-6)
    return xyz.mean(0)-velocity*time.mean(), velocity
