"""Drop-in replacement for `participant_demo/src/algorithm.py` (Official TCP / Docker submission).

Preloads Houyi-Dense-v3 + MasterTracker V14 at module import time (before TCP connection)
so Circle 1 executes in ~27 ms without model-loading overhead.

Configure via environment variables:
  - B02_STARTER_DIR : path to `b02-starter/`
  - B02_HOUYI_DIR   : path to `HouyiBypassV2/`
  - B02_CHECKPOINT  : path to `houyi-dense-v3-e02.pt`
  - B02_DEVICE      : cuda | cpu | auto (default: auto)
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
    # Works whether placed in participant_demo/src/ or participant_demo_v2/participant_demo/src/
    _CANDS = [p / "repo" / "b02-starter" for p in _THIS.parents] + [p / "b02-starter" for p in _THIS.parents]
    _STARTER = next((p for p in _CANDS if p.is_dir()), _THIS.parent)

if str(_STARTER) not in sys.path:
    sys.path.insert(0, str(_STARTER))

from b02.master_tracker import load_tracker  # noqa: E402

_DEVICE = os.environ.get("B02_DEVICE", "auto")
_CKPT = os.environ.get("B02_CHECKPOINT") or None
_HANDLER, _CONFIG = load_tracker(checkpoint=_CKPT, device=_DEVICE)


def process_circle(points, state):
    """Causal scan-by-scan radar target detection & tracking handler."""
    if not points:
        return []
    return _HANDLER(points, state)
