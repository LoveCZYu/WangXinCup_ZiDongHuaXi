"""Validate two official container runs without altering generated result files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_run(root):
    server = json.loads((root / 'run_summary.json').read_text(encoding='utf-8'))
    client = json.loads((root / 'logs/client-summary.json').read_text(encoding='utf-8'))
    if server['status'] != 'success' or client['status'] != 'success':
        raise ValueError(f'Run failed: {root}')
    for key in ('packages', 'circles', 'tracks'):
        if server[key] != client['counts'][key]:
            raise ValueError(f'Official/client count mismatch: {key}')
    for official_packet, client_packet in zip(server['per_package'], client['counts']['per_package'], strict=True):
        if any(official_packet[k] != client_packet[k] for k in ('circles', 'tracks')):
            raise ValueError('Official/client per-packet counts differ')
    if len(client['scans']) != server['circles'] or client['handler_max_ms'] >= 29000:
        raise ValueError('Incomplete or timed-out handler measurements')
    if client['device'] != 'cpu' or client['torch_cuda_initialized']:
        raise ValueError('Frozen deployment must use CPU without CUDA initialization')
    log = (root / 'logs/server.log').read_bytes()
    text = log.decode('utf-8', errors='replace')
    if '收到' not in text:
        text = log.decode('gb18030', errors='replace')
    latencies = [float(s) for s in re.findall(r'耗时([\d.]+)ms', text)]
    if len(latencies) != server['circles'] or max(latencies) >= 30000:
        raise ValueError('Official per-circle timing records incomplete or over limit')
    files = {p['name']: sha(root / p['name']) for p in server['per_package']}
    for package in server['per_package']:
        if not (root / (Path(package['name']).stem + '_rt.dat')).is_file():
            raise ValueError('Official timing output missing')
    return {'status': 'success', 'packages': server['packages'], 'circles': server['circles'],
            'tracks': server['tracks'], 'wall_seconds': server['wall_seconds'],
            'handler_max_ms': client['handler_max_ms'], 'official_max_roundtrip_ms': max(latencies),
            'client_peak_rss_mib': client['client_peak_rss_mib'],
            'container_peak_memory_mib': client.get('container_peak_memory_mib'),
            'package_manifest_sha256': client['package_manifest_sha256'],
            'versions': client['versions'], 'trajectory_sha256': files}


def verify(archive, first, second, report, sample_input):
    with zipfile.ZipFile(archive) as package:
        required = {'Dockerfile', 'run.sh', 'requirements.txt', 'README.md', 'src/main.py', 'PACKAGE.json'}
        names = set(package.namelist())
        if not required <= names:
            raise ValueError('Missing mandatory files at ZIP root')
        manifest = json.loads(package.read('PACKAGE.json'))
        manifest_sha = hashlib.sha256(package.read('PACKAGE.json')).hexdigest()
        if names != set(manifest['files']) | {'PACKAGE.json'}:
            raise ValueError('ZIP file listing differs from manifest')
        for name, expected in manifest['files'].items():
            if hashlib.sha256(package.read(name)).hexdigest() != expected:
                raise ValueError(f'ZIP integrity failure: {name}')
        if any(n.endswith(('.dat', '.csv')) or '__pycache__' in n for n in names):
            raise ValueError('Input or historical data must not be packaged')
    a, b = inspect_run(first), inspect_run(second)
    if any(run['package_manifest_sha256'] != manifest_sha for run in (a, b)):
        raise ValueError('Runtime package does not match the delivered ZIP')
    inputs = {p.name: sha(p) for p in sorted(sample_input.glob('*.dat'))}
    official_sample_sha = '1c16d3c38520f7edaaa1afe6e5196a8e348eb519c7f0d9688467eed2d6b22acc'
    if (not inputs or set(inputs) != set(a['trajectory_sha256'])
            or any(value != official_sample_sha for value in inputs.values())):
        raise ValueError('Sample acceptance requires only the known official train_005 DAT (or copies)')
    if a['trajectory_sha256'] != b['trajectory_sha256']:
        raise ValueError('Repeated runs produced different trajectories')
    value = {'format': 'b02-submission-acceptance-v1', 'team_id': manifest['team_id'],
             'version': manifest['version'], 'algorithm_version': manifest.get('algorithm_version', 'v1'),
             'official_release': manifest.get('official_release'),
             'package': archive.name, 'package_sha256': sha(archive),
             'repeated_trajectory_bytes_identical': True,
             'timing_files_excluded_from_equality': True, 'first': a, 'second': b,
             'input_sha256': inputs,
             'sealed_holdout_used': False,
             'scope': 'Official sample only; not official validation set or website acceptance'}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--sample-input', type=Path, required=True)
    args = parser.parse_args()
    verify(args.zip, args.first, args.second, args.report, args.sample_input)
