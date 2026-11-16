"""Reusable causal inference construction for frozen experimental candidates."""
from pathlib import Path
import json
from experiments.v2_association.run_stage1 import sha

ROOT=Path(__file__).resolve().parents[2]


def load_scorer(checkpoint,digest=None):
    import torch
    from experiments.v2_association.model import AssociationModel,Scorer
    checkpoint=Path(checkpoint)
    if digest is not None:assert sha(checkpoint)==digest
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);meta=saved['metadata']
    if meta['format']=='b02-v3-ownership-training-v1':
        from experiments.v3_tracking.ownership import OwnershipBundle, TRACK_DIM, EDGE_DIM
        for name,expected in meta['runtime_sha256'].items():assert sha(Path(__file__).parent/name)==expected
        prior=saved['prior'];prior_info=prior['metadata']
        assert prior_info['format']=='b02-v3-association-training-v1' and prior_info['memory']
        assert all(prior_info['pairs'][role]==meta['pairs'][role] for role in ('fit','calibration'))
        prior_model=AssociationModel(prior_info['track_dim'],prior_info['edge_dim'])
        prior_model.load_state_dict(prior['model'],strict=True)
        model=AssociationModel(TRACK_DIM,EDGE_DIM);model.load_state_dict(saved['model'],strict=True)
        return OwnershipBundle(Scorer(prior_model,'cpu',1.,'rank_only'),model),True,meta,saved['step']
    if meta['format']=='b02-v3-association-training-v1':
        for name,expected in meta['runtime_sha256'].items():assert sha(Path(__file__).parent/name)==expected
        dimensions=(meta['track_dim'],meta['edge_dim']);memory=meta['memory']
    else:
        assert meta['format']=='b02-residual-association-v1' and meta['feature_version']=='b02-association-context-v2'
        for name in ('features.py','context.py','model.py','tracker.py'):
            assert sha(ROOT/'experiments/v2_association'/name)==meta['sources'][name]
        dimensions=(60,50);memory=False
    model=AssociationModel(*dimensions);model.load_state_dict(saved['model'],strict=True)
    return Scorer(model,'cpu',1.,'rank_only'),memory,meta,saved['step']


class Pipeline:
    def __init__(self,scorer,memory=False,identity='physical'):
        from experiments.v3_tracking.tracker import Tracker
        from experiments.v3_tracking.identity import IdentityManager
        if hasattr(scorer,'build_tracker'):
            if identity!='physical':raise ValueError('Ownership scoring uses the physical track IDs')
            self.physical=scorer.build_tracker()
        else:self.physical=Tracker(scorer,memory=memory)
        if identity=='physical':self.identity=None
        elif identity=='geo4':self.identity=IdentityManager(None,4)
        elif identity in ('learned4','learned8'):self.identity=IdentityManager(scorer,int(identity[-1]),memory)
        else:raise ValueError(identity)

    @property
    def engine(self):return self.identity.engine if self.identity else self.physical

    def __call__(self,points,scores):
        selected=self.physical(points,scores)
        if self.identity:return self.identity(points,scores,selected,self.physical.current_context)
        return selected
