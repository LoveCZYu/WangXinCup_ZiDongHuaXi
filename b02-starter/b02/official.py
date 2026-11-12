"""Use the official demo's executable, server wrapper and TCP client unchanged."""

import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from zipfile import ZipFile

ROOT_IN_ZIP = ("participant_demo_v2", "participant_demo")
SOURCE_FILES = ("algorithm.py", "client.py", "exe_runtime.py", "main.py", "serve.py", "windows_job.py")


def prepare(archive_path, demo_dir):
    required = {("src", name) for name in SOURCE_FILES} | {("data", "official", "serve.exe")}
    with ZipFile(archive_path) as archive:
        selected = {}
        for member in archive.namelist():
            parts = PurePosixPath(member).parts
            if parts[:2] != ROOT_IN_ZIP:
                continue
            local = parts[2:]
            is_sample = (len(local) == 3 and local[:2] == ("data", "inputs")
                         and local[2].startswith("train_") and local[2][6:-4].isdigit()
                         and local[2].endswith(".dat"))
            if local in required or is_sample:
                if local in selected:
                    raise ValueError(f"官方 ZIP 包含重复文件：{member}")
                selected[local] = member
        if not required.issubset(selected) or len(selected) == len(required):
            raise ValueError("ZIP 缺少官方 participant_demo_v2 的程序或样例数据")
        if demo_dir.exists():
            raise ValueError(f"官方程序目录已存在，请选一个新目录：{demo_dir}")
        demo_dir.mkdir(parents=True)
        for local, member in selected.items():
            target = demo_dir.joinpath(*local)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as destination:
                copyfileobj(source, destination)
    print(f"已准备 {len(selected) - len(required)} 个官方样例包：{demo_dir.resolve()}")


def _load_source(demo_dir, module):
    path = demo_dir / "src" / f"{module}.py"
    if not path.is_file():
        raise ValueError(f"缺少官方代码：{path}。请先运行 prepare-official")
    spec = importlib.util.spec_from_file_location(f"b02_official_{module}", path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    return source


def run_official(demo_dir, input_dir, out, port, handler, config, *, original=False):
    demo_dir = demo_dir.resolve()
    input_dir = (input_dir or demo_dir / "data/inputs").resolve()
    out = out.resolve()
    client = _load_source(demo_dir, "client")
    if original:
        handler = _load_source(demo_dir, "algorithm").process_circle
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".b02-session-", dir=out.parent) as temp:
        ready = Path(temp) / "ready"
        status = Path(temp) / "status"
        args = [sys.executable, "-u", str(demo_dir / "src/serve.py"), "--input-dir", str(input_dir),
                "--out-dir", str(out), "--port", str(port), "--task", "all", "--test",
                "--ready-file", str(ready), "--client-status-file", str(status)]
        process = subprocess.Popen(args, env=env)
        try:
            deadline = monotonic() + 150
            while not ready.is_file():
                if process.poll() is not None:
                    raise RuntimeError(f"官方服务未能启动，退出码 {process.returncode}")
                if monotonic() >= deadline:
                    raise TimeoutError("官方服务启动超时")
                sleep(0.05)
            actual_port, packages = map(int, ready.read_text(encoding="ascii").split())
            try:
                counts = client.run_client(handler, packages, "127.0.0.1", actual_port)
            except (Exception, KeyboardInterrupt):
                _write_status(status, 1)
                process.wait(timeout=65)
                raise
            _write_status(status, 0)
            code = process.wait(timeout=65)
            if code != 0:
                raise RuntimeError(f"官方服务运行失败，退出码 {code}。日志目录：{out / 'logs'}")
            print(f"客户端完成 {counts['packages']} 包 / {counts['circles']} 圈 / "
                  f"{counts['points']} 个输入点 / {counts['tracks']} 个输出点")
            if not original:
                (out / "tracker_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _write_status(status, code):
    temp = status.with_suffix(".tmp")
    temp.write_text(f"{code}\n", encoding="ascii")
    temp.replace(status)
