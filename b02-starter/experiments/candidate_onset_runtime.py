"""Opt-in research replay; official v1.1 entry and artifacts remain frozen."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class Handler:
    """Shared immutable models; all point history and tracks are per packet."""
    def __init__(self,point,tracker_factory):
        self.point=point;self.tracker_factory=tracker_factory

    def __call__(self,points,state):
        import torch
        history=state.setdefault('candidate_point_history',deque(maxlen=self.point.history-1))
        tracker=state.get('candidate_tracker')
        if tracker is None:tracker=state['candidate_tracker']=self.tracker_factory()
        with torch.inference_mode():
            scores=self.point.model(torch.as_tensor(self.point.features(points,list(history)),device=self.point.device)).sigmoid().cpu().tolist()
        history.append(points)
        return tracker(points,scores)


def load_handler(resources=ROOT,mode='combined',threads=1):
    import torch
    from b02.point_tracker import load_tracker
    from experiments.candidate_onset.mixed import install,MODES
    from experiments.robust_validation.frozen import Components
    from experiments.v2_association.run_stage1 import sha
    if mode not in MODES:raise ValueError(mode)
    resources=Path(resources).resolve()
    paths={name:resources/path for name,path in dict(prior='runs/v3-tracking/memory-selected.pt',
        affinity='runs/v2-id-recovery/affinity.pt',head='runs/v2-continuation/training/20261116.pt',
        fusion='runs/v2-continuation/fusion-calibration/20261116.json',point='assets/point-scorer-8000.pt').items()}
    missing=[str(path) for path in paths.values() if not path.is_file()]
    if missing:raise ValueError('缺少研究权重，请使用配套便携包或指定 --resources：'+'；'.join(missing))
    protocol=ROOT/'experiments/candidate_onset/mixed-protocol.json'
    plan=json.loads(protocol.read_text(encoding='utf-8'))
    for name,digest in plan['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('冻结候选源码发生变化：'+name)
    components=Components(paths['prior'],paths['affinity'],paths['head'],paths['fusion'],ROOT/'experiments/robust_validation/freeze.json')
    point,info=load_tracker(checkpoint=paths['point'],device='cpu')
    if sha(paths['point'])!=components.freeze['artifacts']['point']:raise ValueError('点模型与冻结候选不一致')
    torch.set_num_threads(threads)
    handler=Handler(point,lambda:install(components.build('candidate'),mode))
    return handler,dict(format='b02-candidate-research-runtime-v1',name='trusted-backup-column-20261117',mode=mode,
        point_seed=info['seed'],point_steps=info['step'],research_only=True,official_release=False,
        weights_sha256={name:sha(path) for name,path in paths.items()},protocol_sha256=sha(protocol))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scene',action='append',required=True,help='明确指定场景，可重复；入口保护正式封存场景')
    p.add_argument('--resources',type=Path,default=ROOT)
    p.add_argument('--mode',choices=('baseline','column','backup','combined'),default='combined')
    args=p.parse_args()
    split=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    if set(args.scene)&set(split['sealed_excluded']):raise ValueError('该研究入口保护正式封存场景，请选择已开发场景')
    if len(args.scene)!=len(set(args.scene)):raise ValueError('场景重复')
    if args.out.exists():raise ValueError('输出目录已存在，请换一个新目录')
    handler,metadata=load_handler(args.resources,args.mode)
    from b02.replay import replay
    replay(args.data,args.out,handler,metadata,args.scene)


if __name__=='__main__':main()
