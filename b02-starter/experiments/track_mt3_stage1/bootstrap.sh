#!/usr/bin/env bash
# Run inside a CUDA PyTorch development image. All project files use persistent storage.
set -eu
base=${1:-/root/shared-nvme/b02}
revision=22c3a19c15df6c560eaadc74462b9b632ede2602
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
mkdir -p "$base/code" "$base/envs" "$base/runs" "$base/scripts"
if [ ! -d "$base/code/CNSF/.git" ]; then
    git clone https://github.com/daihuangyu/CNSF.git "$base/code/CNSF"
fi
cd "$base/code/CNSF"
if [ -n "$(git status --porcelain)" ]; then
    echo 'Upstream checkout contains changes; refusing to change revision.' >&2
    exit 1
fi
git checkout --detach "$revision"
# This platform image lacks ensurepip but already supplies pip and CUDA PyTorch.
python -m venv --without-pip --system-site-packages "$base/envs/trackmt3"
"$base/envs/trackmt3/bin/python" -m pip install --no-deps --no-build-isolation -e .
"$base/envs/trackmt3/bin/python" -c 'import torch, numpy, scipy, yaml, pytest; print("Imports OK; CUDA:", torch.cuda.is_available())'
