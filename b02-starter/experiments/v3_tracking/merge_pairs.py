"""Join disjoint pair-generation reports without opening confirmation data."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--shards',type=int,required=True);a=p.parse_args()
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'));all_scenes=plan['fit64']+plan['calibration'];scenes={};parts=[]
    for i in range(a.shards):
        part=json.loads((a.pairs/f'part-{i}.json').read_text(encoding='utf-8'))
        assert part['complete'] and part['shard_index']==i and part['shard_count']==a.shards
        assert set(part['scenes'])==set(all_scenes[i::a.shards]) and not set(scenes)&set(part['scenes'])
        scenes.update(part['scenes']);parts.append(part)
    assert len({p['prefix'] for p in parts})==1 and len({p['data_sha256'] for p in parts})==1
    assert set(scenes)==set(all_scenes)
    manifest={'format':'b02-v3-pair-dataset-v1','fit':plan['fit64'],'calibration':plan['calibration'],'confirmation':[],'dev':[],'holdout':[],
        'prefix':parts[0]['prefix'],'context':True,'feature_version':'b02-association-context-v2','track_dim':60,'edge_dim':50,'max_edges':12,
        'memory_available':parts[0]['prefix']=='v2','memory_feature_version':'b02-association-memory-v3','memory_track_dim':98,'memory_edge_dim':85,
        'data_sha256':parts[0]['data_sha256'],'cache_manifest_sha256':parts[0]['cache_manifest_sha256'],'split_sha256':parts[0]['split_sha256'],
        'frame_selection':'8 evenly spaced 8-scan clips per file; causal complete prefixes',
        'truth_usage':'supervision only; pure history and causal first-reliable-anchor labels; never model inputs'}
    assert not (a.pairs/'manifest.json').exists();save(a.pairs/'manifest.json',manifest)
    save(a.pairs/'report.json',{'complete':True,'scenes':scenes,'part_wall_seconds':[p['wall_seconds'] for p in parts]})
    print(json.dumps({'pairs_complete':True,'scenes':len(scenes),'prefix':manifest['prefix']}),flush=True)


if __name__=='__main__':main()
