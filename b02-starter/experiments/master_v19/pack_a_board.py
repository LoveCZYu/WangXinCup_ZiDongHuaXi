"""Run official serve.exe on A-board .dat inputs and package the submission ZIP for MasterTracker V19.5.

Usage (Windows, from `b02-starter/`):
  python experiments/master_v19/pack_a_board.py \
      --demo-dir "path/to/participant_demo" \
      --out-dir runs/a_board_master_v19 \
      --zip-out runs/B02_A_master_v19.zip \
      --houyi-dir "path/to/HouyiBypassV2" \
      --checkpoint "path/to/houyi-dense-v3-e02.pt,path/to/houyi-v18-joint-12scan.pt" \
      --pair-checkpoint "path/to/houyi-v18-joint-12scan.pt" \
      --seq-checkpoint "path/to/track_seq_cross_attn_v18.pt"
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def main():
    parser = argparse.ArgumentParser(description="Run official serve.exe TCP client and package A-board ZIP.")
    parser.add_argument("--demo-dir", type=Path, required=True, help="Unpacked participant_demo directory")
    parser.add_argument("--input-dir", type=Path, default=None, help="Defaults to <demo-dir>/data/inputs")
    parser.add_argument("--out-dir", type=Path, required=True, help="Fresh output directory for serve.exe")
    parser.add_argument("--zip-out", type=Path, required=True, help="Target A-board submission ZIP path")
    parser.add_argument("--houyi-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single .pt path or comma-separated list (e.g. v3.pt,v18.pt)")
    parser.add_argument("--pair-checkpoint", type=Path, default=None)
    parser.add_argument("--seq-checkpoint", type=Path, default=None)
    parser.add_argument("--no-az-tta", action="store_true", help="Disable 50%%-overlap azimuth sector TTA")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=19902)
    args = parser.parse_args()

    demo = args.demo_dir.resolve()
    input_dir = (args.input_dir or (demo / "data" / "inputs")).resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists (must be empty): {out_dir}")
    tmp_dir = out_dir.parent / (out_dir.name + "_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    ready_file = tmp_dir / "ready.txt"
    status_file = tmp_dir / "client_status.txt"

    starter_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["B02_STARTER_DIR"] = str(starter_root)
    env["B02_DEVICE"] = args.device
    if args.houyi_dir:
        env["B02_HOUYI_DIR"] = str(args.houyi_dir.resolve())
    if args.checkpoint:
        resolved_ckpts = [str(Path(s.strip()).resolve()) for s in args.checkpoint.split(",") if s.strip()]
        env["B02_CHECKPOINT"] = ",".join(resolved_ckpts)
    if args.pair_checkpoint:
        env["B02_PAIR_CHECKPOINT"] = str(args.pair_checkpoint.resolve())
    if args.seq_checkpoint:
        env["B02_SEQ_CHECKPOINT"] = str(args.seq_checkpoint.resolve())
    if args.no_az_tta:
        env["B02_AZ_TTA"] = "0"

    srv_cmd = [
        sys.executable, str(demo / "src" / "serve.py"),
        "--input-dir", str(input_dir),
        "--out-dir", str(out_dir),
        "--port", str(args.port),
        "--ready-file", str(ready_file),
        "--client-status-file", str(status_file),
    ]
    srv = subprocess.Popen(srv_cmd, cwd=str(demo), env=env)
    deadline = time.monotonic() + 60
    while not ready_file.exists():
        if srv.poll() is not None:
            raise RuntimeError(f"serve.py exited early with code {srv.returncode}")
        if time.monotonic() > deadline:
            srv.terminate()
            raise TimeoutError("Timed out waiting for serve.py ready file")
        time.sleep(0.1)

    port_str, expected_str = ready_file.read_text(encoding="ascii").strip().split()
    cli_cmd = [
        sys.executable, str(demo / "src" / "main.py"),
        "--host", "127.0.0.1",
        "--port", port_str,
        "--expected-packages", expected_str,
    ]
    cli = subprocess.run(cli_cmd, cwd=str(demo), env=env)
    status_file.write_text(f"{cli.returncode}\n", encoding="ascii")
    srv_code = srv.wait(timeout=60)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    if cli.returncode != 0 or srv_code != 0 or summary.get("status") != "success":
        raise RuntimeError(f"Official TCP run failed: client={cli.returncode} server={srv_code} summary={summary}")

    args.zip_out.parent.mkdir(parents=True, exist_ok=True)
    input_names = sorted(p.name for p in input_dir.glob("*.dat"))
    with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in input_names:
            zf.write(out_dir / name, arcname=name)
    print(f"[OK] Packed {len(input_names)} encrypted .dat files -> {args.zip_out} ({args.zip_out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
