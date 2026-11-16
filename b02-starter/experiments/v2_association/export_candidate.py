"""Export only the preregistered seed as a portable candidate inference artifact."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--weights-out',type=Path,required=True);p.add_argument('--profile-out',type=Path,required=True)
    a=p.parse_args()
    import torch
    freeze=json.loads(a.frozen.read_text(encoding='utf-8'));assert freeze['checks_passed']
    seed=freeze['protocol']['production_seed_if_validated'];item=next(x for x in freeze['models'] if x['seed']==seed)
    assert sha(a.checkpoint)==item['sha256']
    loaded=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert loaded['step']==item['step'] and loaded['metadata']['seed']==seed
    assert loaded['metadata']['feature_version']==freeze['feature_version']
    for name,digest in freeze['runtime_feature_sha256'].items():assert sha(Path(__file__).parent/name)==digest
    assert sha(ROOT/'profiles/current.json')==freeze['profile_sha256']
    assert sha(ROOT/'assets/point-scorer-8000.pt')==freeze['point_model_sha256']
    assert not a.weights_out.exists() and not a.profile_out.exists()
    a.weights_out.parent.mkdir(parents=True,exist_ok=True)
    metadata={'format':'b02-association-inference-v1','feature_version':freeze['feature_version'],
              'seed':seed,'step':loaded['step'],'training_checkpoint_sha256':sha(a.checkpoint)}
    torch.save({'metadata':metadata,'model':loaded['model']},a.weights_out)
    spec={'format':'b02-association-profile-v1','name':'v2-context-candidate','status':'experimental_candidate',
          'feature_version':freeze['feature_version'],'seed':seed,'step':loaded['step'],
          'checkpoint':Path(os.path.relpath(a.weights_out.resolve(),a.profile_out.parent.resolve())).as_posix(),
          'checkpoint_sha256':sha(a.weights_out),'point_profile':Path(os.path.relpath(ROOT/'profiles/current.json',a.profile_out.parent.resolve())).as_posix(),
          'point_profile_sha256':freeze['profile_sha256'],'strategy':freeze['strategy'],'scale':freeze['scale'],
          'runtime_feature_sha256':freeze['runtime_feature_sha256'],'freeze_sha256':sha(a.frozen),
          'official_submission_ready':False,'official_score':None}
    save(a.profile_out,spec);print(json.dumps({'profile':str(a.profile_out),'weights_bytes':a.weights_out.stat().st_size,
                                           'checkpoint_sha256':spec['checkpoint_sha256']}),flush=True)


if __name__=='__main__':main()
