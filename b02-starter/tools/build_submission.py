"""Build an isolated deterministic submission ZIP; never mutate frozen v1 sources."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'submission-template'
OFFICIAL_FILES = ('client.py', 'serve.py', 'exe_runtime.py', 'windows_job.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output, team_id='T000906', official=None, version='v1.1'):
    if not re.fullmatch(r'T\d{6}', team_id):
        raise ValueError('team_id must have form T000906')
    if team_id != 'T000906':
        raise ValueError('This frozen template is assigned to T000906')
    if not re.fullmatch(r'v\d+(?:\.\d+)*', version):
        raise ValueError('version must have form v1.1')
    official = Path(official or ROOT / '.official')
    frozen = json.loads((TEMPLATE / 'frozen-v1.json').read_text(encoding='utf-8'))
    # Validate all prerequisites before creating output.
    for name, sha in frozen['solution_files'].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f'Frozen v1 source changed: {name}')
    sources = {f'src/{name}': official / 'src' / name for name in OFFICIAL_FILES}
    sources['data/official/serve.exe'] = official / 'data/official/serve.exe'
    sources['run.sh'] = TEMPLATE / 'run.sh'
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, expected in frozen['official_files'].items():
        if digest(sources[name]) != expected:
            raise ValueError(f'Official runtime differs from frozen sample: {name}')
    output = Path(output).resolve()
    stage = output / f'{team_id}_{version}_code'
    archive = output / f'{stage.name}.zip'
    if stage.exists() or archive.exists():
        raise FileExistsError(f'Refusing to overwrite existing build: {stage}')
    stage.mkdir(parents=True)
    for path in TEMPLATE.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        target = stage / path.relative_to(TEMPLATE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for name in frozen['solution_files']:
        target = stage / 'src/solution' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    for name, source in sources.items():
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Keep official lifecycle management; only route our extra log to its out-dir.
    run = (stage / 'run.sh').read_text(encoding='utf-8')
    launch = 'python3 -u src/serve.py "$@"'
    if run.count(launch) != 1:
        raise ValueError('Official run.sh launch line changed')
    run = run.replace(launch, 'python3 -u src/launch_server.py "$@"', 1)
    hook = '''
# Forward the official output argument to the participant's diagnostic logger.
export B02_OUTPUT_DIR=./output
arguments=("$@")
for ((i=0; i<${#arguments[@]}; i++)); do
  if [[ "${arguments[i]}" == "--out-dir" && $((i+1)) -lt ${#arguments[@]} ]]; then
    export B02_OUTPUT_DIR="${arguments[i+1]}"
  elif [[ "${arguments[i]}" == --out-dir=* ]]; then
    export B02_OUTPUT_DIR="${arguments[i]#--out-dir=}"
  fi
done
'''
    run = run.replace('session_dir=$(mktemp -d)', hook + '\nsession_dir=$(mktemp -d)', 1)
    (stage / 'run.sh').write_text(run, encoding='utf-8', newline='\n')
    manifest = {'format': 'b02-submission-v1', 'team_id': team_id, 'version': version,
                'algorithm_version': 'v1', 'official_release': frozen['official_release'],
                'algorithm_source_commit': '6cd6bdb', 'inference_device': 'cpu',
                'official_original_sha256': {k: digest(v) for k, v in sources.items()},
                'files': {str(p.relative_to(stage)).replace('\\', '/'): digest(p)
                          for p in sorted(stage.rglob('*'), key=lambda p: p.relative_to(stage).as_posix()) if p.is_file()}}
    (stage / 'PACKAGE.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for path in sorted(stage.rglob('*'), key=lambda p: p.relative_to(stage).as_posix()):
            if path.is_file():
                name = path.relative_to(stage).as_posix()
                info = zipfile.ZipInfo(name, (2026, 11, 15, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100755 if path.suffix == '.sh' else 0o100644) << 16
                package.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    result = {'directory': str(stage), 'zip': str(archive), 'sha256': digest(archive),
              'files': len(manifest['files']) + 1, 'bytes': archive.stat().st_size}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--team-id', default='T000906')
    parser.add_argument('--official', type=Path)
    parser.add_argument('--version', default='v1.1')
    args = parser.parse_args()
    build(args.out, args.team_id, args.official, args.version)
