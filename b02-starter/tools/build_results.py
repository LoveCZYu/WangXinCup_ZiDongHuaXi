"""Verify two frozen A-board runs and ZIP only the five official trajectory DATs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from verify_submission import inspect_run, sha

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'splits/a_board.json'
EXPECTED = {f'train_{i:03d}.dat' for i in range(1, 6)}


def code_manifest(archive):
    with zipfile.ZipFile(archive) as source:
        raw = source.read('PACKAGE.json')
        manifest = json.loads(raw)
        if set(source.namelist()) != set(manifest['files']) | {'PACKAGE.json'}:
            raise ValueError('Code ZIP listing differs from its manifest')
        for name, expected in manifest['files'].items():
            if hashlib.sha256(source.read(name)).hexdigest() != expected:
                raise ValueError(f'Code ZIP integrity failure: {name}')
    return manifest, hashlib.sha256(raw).hexdigest()


def build(code_zip, input_dir, first, repeat, output, report, policy_path=POLICY):
    paths = [Path(p).resolve() for p in (code_zip, input_dir, first, repeat, output, report)]
    code_zip, input_dir, first, repeat, output, report = paths
    if output.exists() or report.exists():
        raise FileExistsError('Refusing to replace an existing result ZIP or report')
    if output == report or first == repeat:
        raise ValueError('Use two separate runs, and separate output/report files')
    policy = json.loads(Path(policy_path).read_text(encoding='utf-8'))
    if set(policy['inputs']) != EXPECTED or set(policy['result_zip_files']) != EXPECTED:
        raise ValueError('A-board policy must identify exactly train_001 to train_005')
    if {p.name for p in input_dir.glob('*.dat')} != EXPECTED:
        raise ValueError('Input directory must contain exactly the five official A-board files')
    inputs = {name: sha(input_dir/name) for name in sorted(EXPECTED)}
    if any(inputs[name] != policy['inputs'][name]['sha256'] for name in EXPECTED):
        raise ValueError('Input DAT does not match the confirmed official release')
    manifest, manifest_sha = code_manifest(code_zip)
    a, b = inspect_run(first), inspect_run(repeat)
    for run, folder in ((a, first), (b, repeat)):
        if run['packages'] != 5 or set(run['trajectory_sha256']) != EXPECTED:
            raise ValueError('Official run did not finish exactly the five A-board packets')
        if run['package_manifest_sha256'] != manifest_sha:
            raise ValueError('Runtime package does not match the delivered code ZIP')
        official = json.loads((folder/'run_summary.json').read_text(encoding='utf-8'))
        run['per_package'] = official['per_package']
    if a['trajectory_sha256'] != b['trajectory_sha256']:
        raise ValueError('Repeated A-board runs produced different trajectory bytes')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+'.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(EXPECTED):
                data = (first/name).read_bytes()
                if hashlib.sha256(data).hexdigest() != a['trajectory_sha256'][name]:
                    raise ValueError(f'Trajectory changed during packaging: {name}')
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != EXPECTED:
                raise ValueError('Result ZIP is corrupt or contains unexpected files')
            if any(hashlib.sha256(archive.read(n)).hexdigest() != a['trajectory_sha256'][n] for n in EXPECTED):
                raise ValueError('Result ZIP bytes differ from official trajectory outputs')
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    value = {'format':'b02-a-board-delivery-v1', 'date':policy['date'],
             'team_id':manifest['team_id'], 'version':manifest['version'],
             'code_zip':code_zip.name, 'code_zip_sha256':sha(code_zip),
             'package_manifest_sha256':manifest_sha,
             'result_zip':output.name, 'result_zip_sha256':sha(output),
             'result_zip_bytes':output.stat().st_size, 'result_files':sorted(EXPECTED),
             'input_sha256':inputs, 'first':a, 'repeat':b,
             'all_trajectories_reproduced_exactly':True,
             'timing_files_and_logs_excluded_from_upload':True,
             'historical_holdout_overlap':policy.get('historical_holdout_overlap', []),
             'remaining_independent_holdout_count':len(policy.get('remaining_independent_holdout', [])),
             'csv_truth_scoring_performed':False, 'training_or_tuning_performed':False,
             'website_uploaded':False, 'official_score':None,
             'organizer_clarification':policy.get('authority')}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'result_zip':str(output),'sha256':value['result_zip_sha256'],
                      'files':value['result_files'],'circles':a['circles'],'tracks':a['tracks']},
                     ensure_ascii=False,indent=2))
    return value


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-zip',type=Path,required=True)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--repeat',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    build(args.code_zip,args.input_dir,args.run,args.repeat,args.out,args.report)
