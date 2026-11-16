"""Verify exact cache role coverage and create a single immutable cache manifest."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--shards',type=int,required=True);p.add_argument('--catalogue',type=Path,required=True);a=p.parse_args()
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'));scenes={};parts=[]
    for i in range(a.shards):
        part=json.loads((a.cache/f'part-{i}.json').read_text(encoding='utf-8'))
        assert part['complete'] and part['shard_index']==i and part['shard_count']==a.shards
        assert set(part['scenes'])==set((plan['fit64']+plan['calibration'])[i::a.shards])
        assert not set(scenes)&set(part['scenes']);scenes.update(part['scenes']);parts.append(part)
    assert set(scenes)==set(plan['fit64']+plan['calibration'])
    assert len({p['point_weights_sha256'] for p in parts})==1
    path=a.cache/'manifest.json';assert not path.exists()
    save(path,{'format':'b02-v3-point-cache-v1','complete':True,'fit':plan['fit64'],'calibration':plan['calibration'],
         'confirmation':[],'dev':[],'holdout':[],'data_sha256':parts[0]['data_sha256'],
         'point_profile_sha256':parts[0]['point_profile_sha256'],'point_weights_sha256':parts[0]['point_weights_sha256'],
         'split_sha256':sha(ROOT/'splits/v3_tracking.json'),'catalogue_sha256':sha(a.catalogue),'scenes':scenes})
    print(json.dumps({'cache_complete':True,'scenes':len(scenes)}),flush=True)


if __name__=='__main__':main()
