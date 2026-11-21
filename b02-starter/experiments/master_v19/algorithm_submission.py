"""Drop-in replacement for `participant_demo/src/algorithm.py` (Official TCP / Docker submission).

Preloads Houyi-Dense-v3 + Houyi-V18-Joint-12scan + TrackSeqCrossAttnNet + MasterTracker V19.5
at module import time (before TCP connection) so Circle 1 executes without model-loading overhead.

Configure via environment variables:
  - B02_STARTER_DIR     : path to `b02-starter/`
  - B02_HOUYI_DIR       : path to `HouyiBypassV2/`
  - B02_CHECKPOINT      : single `.pt` or comma-separated `v3.pt,v18.pt`
  - B02_PAIR_CHECKPOINT : path to `houyi-v18-joint-12scan.pt` (reused when also in B02_CHECKPOINT)
  - B02_SEQ_CHECKPOINT  : path to `track_seq_cross_attn_v18.pt`
  - B02_WEIGHTS         : optional comma-separated ensemble weights (default: 0.30,0.70)
  - B02_TEMPERATURE     : optional logit temperature (default: 0.90 for ensemble)
  - B02_AZ_TTA          : `1` (default) or `0` to disable 50%-overlap azimuth sector TTA
  - B02_DEVICE          : cuda | cpu | auto (default: auto)
"""
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_THIS = Path(__file__).resolve()
_STARTER_ENV = os.environ.get("B02_STARTER_DIR")
if _STARTER_ENV:
    _STARTER = Path(_STARTER_ENV)
else:
    _CANDS = [p / "repo" / "b02-starter" for p in _THIS.parents] + [p / "b02-starter" for p in _THIS.parents]
    _STARTER = next((p for p in _CANDS if p.is_dir()), _THIS.parent)

if str(_STARTER) not in sys.path:
    sys.path.insert(0, str(_STARTER))

from b02.master_tracker import load_tracker  # noqa: E402

_DEVICE = os.environ.get("B02_DEVICE", "auto")
_CKPT = os.environ.get("B02_CHECKPOINT") or None
_PAIR = os.environ.get("B02_PAIR_CHECKPOINT", "auto")
_SEQ = os.environ.get("B02_SEQ_CHECKPOINT", "auto")
_HANDLER, _CONFIG = load_tracker(
    checkpoint=_CKPT,
    pair_checkpoint=_PAIR,
    seq_checkpoint=_SEQ,
    device=_DEVICE,
)


def process_circle(points, state):
    """Causal scan-by-scan radar target detection & tracking handler."""
    if not points:
        return []
    return _HANDLER(points, state)
