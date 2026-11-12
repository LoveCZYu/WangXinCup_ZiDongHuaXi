#!/usr/bin/env bash
set -euo pipefail
base=/root/shared-nvme/b02
python="$base/envs/trackmt3/bin/python"
project="$base/code/b02-stage2"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
"$python" "$project/experiments/track_mt3_stage2/run.py" \
  --upstream "$base/code/CNSF-b02-stage2" --project "$project" \
  --data "$base/data/b02-stage2-pilot.zip" \
  --pretrained "$base/runs/upstream-recipe-10k01/train/checkpoints/best-validation.pt" \
  --out "$base/runs/b02-stage2-pilot02" --steps 1000 --seconds 1200
