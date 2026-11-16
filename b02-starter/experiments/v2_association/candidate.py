"""Portable experimental inference entry; does not change the production v1 entry."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


class CandidateHandler:
    """Shared weights, but all packet history and IDs live in the supplied state."""
    def __init__(self,point_handler,scorer):
        self.point=point_handler;self.scorer=scorer

    def __call__(self,points,state):
        import torch
        from experiments.v2_association.tracker import Tracker
        history=state.setdefault('v2_point_history',deque(maxlen=self.point.history-1))
        tracker=state.get('v2_tracker')
        if tracker is None:tracker=state['v2_tracker']=Tracker('v1',scorer=self.scorer,contextual=True)
        with torch.no_grad():
            scores=self.point.model(torch.as_tensor(self.point.features(points,list(history)),device=self.point.device)).sigmoid().cpu().tolist()
        history.append(points)
        return tracker(points,scores)


def load_candidate(profile,device='cpu'):
    import torch
    from b02.point_tracker import load_tracker,sha256
    from experiments.v2_association.context import FEATURE_VERSION,TRACK_DIM,EDGE_DIM
    from experiments.v2_association.model import AssociationModel,Scorer
    path=Path(profile).resolve();spec=json.loads(path.read_text(encoding='utf-8-sig'))
    if spec.get('format')!='b02-association-profile-v1' or spec['feature_version']!=FEATURE_VERSION:
        raise ValueError('Association profile/feature version mismatch')
    for name,digest in spec['runtime_feature_sha256'].items():
        if sha256(ROOT/'experiments/v2_association'/name)!=digest:raise ValueError(f'Runtime source changed: {name}')
    checkpoint=(path.parent/spec['checkpoint']).resolve();point_profile=(path.parent/spec['point_profile']).resolve()
    if sha256(checkpoint)!=spec['checkpoint_sha256'] or sha256(point_profile)!=spec['point_profile_sha256']:
        raise ValueError('Candidate weights or point profile changed')
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=True);metadata=loaded['metadata']
    for key in ('seed','step','feature_version'):
        if metadata[key]!=spec[key]:raise ValueError(f'Candidate metadata mismatch: {key}')
    if metadata.get('format')!='b02-association-inference-v1':raise ValueError('Expected an inference artifact')
    point,point_spec=load_tracker(point_profile,device=device)
    model=AssociationModel(TRACK_DIM,EDGE_DIM);model.load_state_dict(loaded['model'],strict=True)
    scorer=Scorer(model,device=point.device,scale=spec['scale'],strategy=spec['strategy'])
    return CandidateHandler(point,scorer),{**spec,'profile':str(path),'point_model':point_spec,'device':str(point.device)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--scene',nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');a=p.parse_args()
    split=json.loads((ROOT/'splits/v1.json').read_text(encoding='utf-8'))
    if not set(a.scene)<=set(split['train']+split['dev']):raise ValueError('This preview entry excludes sealed scenes')
    handler,spec=load_candidate(a.profile,a.device)
    from b02.replay import replay
    replay(a.data,a.out,handler,spec,a.scene)


if __name__=='__main__':main()
