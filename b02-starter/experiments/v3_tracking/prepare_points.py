"""Cache frozen point scores on declared fit/calibration scenes; reserve confirmation."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path,PurePosixPath
import sys
from time import monotonic
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--catalogue',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--reuse',type=Path,required=True)
    p.add_argument('--reuse-data',type=Path,required=True)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--seconds',type=int,default=3500);a=p.parse_args()
    import numpy as np
    import torch
    from b02.replay import open_scenes,_scans
    from b02.point_tracker import load_tracker
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    catalogue=json.loads(a.catalogue.read_text(encoding='utf-8'));assert sha(a.data)==catalogue['archive_sha256']
    assert catalogue['roles']=={'fit64':plan['fit64'],'calibration':plan['calibration']}
    names=plan['fit64']+plan['calibration'];assert not set(names)&set(plan['confirmation16']+plan['development_regression']+plan['sealed_excluded'])
    assert 0<=a.shard_index<a.shard_count<=len(names)
    selected=names[a.shard_index::a.shard_count];a.out.mkdir(parents=True,exist_ok=True);start=monotonic()
    reuse_manifest=json.loads((a.reuse/'manifest.json').read_text(encoding='utf-8'))
    assert reuse_manifest['data_sha256']==sha(a.reuse_data)
    handler,profile=load_tracker(device='cpu')
    assert reuse_manifest['profile']['checkpoint_sha256']==profile['checkpoint_sha256']
    report={'complete':False,'shard_index':a.shard_index,'shard_count':a.shard_count,'scenes':{},
            'data_sha256':catalogue['archive_sha256'],'point_profile_sha256':sha(ROOT/'profiles/current.json'),
            'point_weights_sha256':sha(ROOT/'assets/point-scorer-8000.pt'),'split_sha256':sha(ROOT/'splits/v3_tracking.json')}
    with ZipFile(a.reuse_data) as old_archive,open_scenes(a.data) as (_,opener):
        index={PurePosixPath(n).stem:n for n in old_archive.namelist() if n.endswith('.csv')}
        for scene in selected:
            if monotonic()-start>a.seconds:raise TimeoutError('Point cache budget')
            role='fit' if scene in plan['fit64'] else 'calibration';folder=a.out/role;folder.mkdir(exist_ok=True)
            reused=False
            old_role='fit' if scene in reuse_manifest['fit'] else 'calibration' if scene in reuse_manifest['calibration'] else None
            if old_role:
                assert hashlib.sha256(old_archive.read(index[scene])).hexdigest()==catalogue['sources'][scene]
                with np.load(a.reuse/old_role/f'{scene}.npz') as previous:
                    scores=previous['point_scores'];offsets=previous['point_offsets']
                reused=True
            else:
                history=deque(maxlen=5);parts=[];offsets=[0]
                with opener(scene) as file:
                    for circle,points,labels in _scans(file):
                        with torch.no_grad():part=handler.model(torch.as_tensor(handler.features(points,list(history)))).sigmoid().tolist()
                        history.append(points);parts.extend(part);offsets.append(len(parts))
                scores=np.asarray(parts,dtype=np.float32);offsets=np.asarray(offsets,dtype=np.int64)
            assert np.isfinite(scores).all() and offsets[-1]==len(scores)
            path=folder/f'{scene}.npz';assert not path.exists()
            np.savez_compressed(path,point_scores=scores,point_offsets=offsets)
            report['scenes'][scene]={'role':role,'reused':reused,'frames':len(offsets)-1,'points':len(scores),
                'csv_sha256':catalogue['sources'][scene],'cache_sha256':sha(path)}
            save(a.out/f'part-{a.shard_index}.json',report)
            print(json.dumps({'scene':scene,'reused':reused,'seconds':round(monotonic()-start,2)}),flush=True)
    report['complete']=True;report['wall_seconds']=monotonic()-start;save(a.out/f'part-{a.shard_index}.json',report)


if __name__=='__main__':main()
