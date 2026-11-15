"""Frozen v1 adapter. Only the unchanged official server reads input DAT files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'solution'))


def verify_package(root=ROOT):
    manifest = json.loads((root / 'PACKAGE.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Invalid manifest path: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Package integrity check failed: {name}')
    return manifest


class MeasuredHandler:
    def __init__(self, handler):
        self.handler = handler
        self.scans = []
        self.packet = 0
        self.packet_state = None

    def __call__(self, points, state):
        if state is not self.packet_state:
            self.packet += 1
            self.packet_state = state
        started = time.perf_counter()
        tracks = self.handler(points, state)
        self.scans.append({'packet': self.packet, 'circle': points[0]['CircleID'] if points else None,
                           'points': len(points), 'tracks': len(tracks),
                           'handler_ms': 1000 * (time.perf_counter() - started)})
        return tracks


def main(argv=None):
    parser = argparse.ArgumentParser(description='T000906 frozen v1 official TCP client')
    parser.add_argument('--host', default=os.environ.get('SERVER_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=os.environ.get('SERVER_PORT', '9902'))
    parser.add_argument('--expected-packages', type=int, default=os.environ.get('EXPECTED_PACKAGES'))
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.expected_packages is None or args.expected_packages < 1:
        parser.error('Use run.sh, or supply a valid port and positive expected-packages')
    report = {'status': 'failed', 'team_id': 'T000906', 'version': 'v1', 'device': 'cpu'}
    measured = None
    started = time.perf_counter()
    try:
        manifest = verify_package()
        report['package_manifest_sha256'] = hashlib.sha256((ROOT / 'PACKAGE.json').read_bytes()).hexdigest()
        import numpy
        import scipy
        import torch
        from b02.point_tracker import load_tracker
        from client import run_client
        torch.use_deterministic_algorithms(True)
        handler, metadata = load_tracker(device='cpu')
        with torch.no_grad():
            handler.model(torch.zeros((1, 76)))
        measured = MeasuredHandler(handler)
        report.update(profile=metadata, package_version=manifest['version'],
                      versions={'python': sys.version, 'numpy': numpy.__version__,
                                'scipy': scipy.__version__, 'torch': torch.__version__,
                                'cuda_runtime': torch.version.cuda},
                      torch_cuda_initialized=torch.cuda.is_initialized())
        print('[v1] integrity OK; frozen profile; device=cpu; starting official TCP client', flush=True)
        report['counts'] = run_client(measured, args.expected_packages, args.host, args.port)
        report['status'] = 'success'
        return_code = 0
    except (Exception, KeyboardInterrupt) as error:
        report['error'] = f'{type(error).__name__}: {error}'
        print(f"[v1] {report['error']}", file=sys.stderr, flush=True)
        return_code = 1
    report['client_wall_seconds'] = time.perf_counter() - started
    report['client_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    peak = Path('/sys/fs/cgroup/memory.peak')
    report['container_peak_memory_mib'] = int(peak.read_text()) / 1024**2 if peak.is_file() else None
    report['scans'] = measured.scans if measured else []
    report['handler_max_ms'] = max((s['handler_ms'] for s in report['scans']), default=0)
    output = Path(os.environ.get('B02_OUTPUT_DIR', '/output')) / 'logs'
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / 'client-summary.json.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output / 'client-summary.json')
    print(f"[v1] {report['status']}; max handler {report['handler_max_ms']:.1f} ms; "
          f"peak client RSS {report['client_peak_rss_mib']:.1f} MiB", flush=True)
    return return_code


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
