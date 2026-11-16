"""Freeze a two-seed candidate only after preregistered full calibration checks."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    import torch
    assert not a.out.exists()
    protocol=json.loads(a.protocol.read_text(encoding='utf-8'));stage=json.loads((a.runs/'stage1-01/report.json').read_text(encoding='utf-8'))
    pairs=a.runs/'pairs03-context';manifest=json.loads((pairs/'manifest.json').read_text(encoding='utf-8'))
    baseline=stage['results']['v1'];checks={};models=[]
    for seed in protocol['seeds']:
        folder=a.runs/f'train04-{seed}';selection=json.loads((folder/'selected.json').read_text(encoding='utf-8'))
        selected=selection['selected']['step'];checkpoint=folder/f'step-{selected:05d}.pt'
        loaded=torch.load(checkpoint,map_location='cpu')
        assert loaded['step']==selected and loaded['metadata']['seed']==seed
        for name in ('features.py','context.py','model.py','tracker.py'):
            assert loaded['metadata']['sources'][name]==sha(Path(__file__).parent/name)
        history=[json.loads(line) for line in (folder/'history.jsonl').read_text(encoding='utf-8').splitlines()]
        assert selected==min(history,key=lambda row:row['calibration_loss'])['step'] and selected>0
        report_path=a.runs/f'full04-{seed}'/'report.json';r=json.loads(report_path.read_text(encoding='utf-8'))
        assert r['complete'] and r['scenes']==manifest['calibration']
        assert r['strategy']==protocol['strategy'] and r['scale']==protocol['scale']
        assert r['baseline']==baseline['metrics']
        m,b=r['metrics'],baseline['metrics'];c=protocol['feasible_per_seed']
        checks[str(seed)]={
            'recall':m['point_recall']>=b['point_recall']-c['recall_max_drop_pp'],
            'clutter':m['clutter_outputs']<=b['clutter_outputs']*c['clutter_ratio_max'],
            'continuous':m['continuous_score']>=b['continuous_score']+c['continuous_min_gain'],
            'switches':r['identity']['switches']<=baseline['identity']['switches']*c['switch_ratio_max'],
            'weighted_total':m['weighted_total']>=b['weighted_total']+c['weighted_total_min_gain']}
        models.append({'seed':seed,'step':selected,'checkpoint':str(checkpoint),
                       'sha256':sha(checkpoint),'calibration_report_sha256':sha(report_path),
                       'calibration_metrics':m,'calibration_identity':r['identity']})
    audit={'checks':checks,'passed':all(all(c.values()) for c in checks.values())}
    save(a.out.with_suffix('.checks.json'),audit)
    if not audit['passed']:raise ValueError('Full calibration feasibility failed; development remains unopened')
    files=['features.py','context.py','model.py','tracker.py']
    freeze={'format':'b02-v2-candidate-freeze-v1','protocol':protocol,'protocol_sha256':sha(a.protocol),
            'models':models,'checks':checks,'checks_passed':True,
            'data_sha256':manifest['data_sha256'],'pairs':str(pairs),'pairs_manifest_sha256':sha(pairs/'manifest.json'),
            'feature_version':manifest['feature_version'],'strategy':protocol['strategy'],'scale':protocol['scale'],
            'runtime_feature_sha256':{name:sha(Path(__file__).parent/name) for name in files},
            'point_model_sha256':sha(ROOT/'assets/point-scorer-8000.pt'),
            'profile_sha256':sha(ROOT/'profiles/current.json'),
            'dev_scenes':json.loads((ROOT/'splits/v1.json').read_text(encoding='utf-8'))['dev'],
            'holdout_used':False,'official_submission':False}
    save(a.out,freeze);print(json.dumps({'frozen':str(a.out),'checks':checks}),flush=True)


if __name__=='__main__':main()
