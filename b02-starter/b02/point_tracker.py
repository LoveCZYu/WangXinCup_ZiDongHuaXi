"""Shared point-model loader and causal handler for replay and official TCP."""
from collections import deque
import hashlib
import json
from math import isfinite
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
DEFAULT_PROFILE=ROOT/'profiles/current.json'


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


class Handler:
    """Model is shared; each packet's history and tracks live entirely in state."""
    def __init__(self,model,variant,policy):
        from experiments.association.interventions import Tracker
        from experiments.point_scorer.model import HISTORY,point_features
        self.model=model.eval();self.variant=variant;self.policy=policy
        self.device=next(model.parameters()).device
        self.tracker_type=Tracker;self.features=point_features;self.history=HISTORY

    def __call__(self,points,state):
        import torch
        history=state.setdefault('point_history',deque(maxlen=self.history-1))
        tracker=state.get('point_tracker')
        if tracker is None:
            tracker=state['point_tracker']=self.tracker_type(self.variant,self.policy)
        with torch.no_grad():
            scores=self.model(torch.as_tensor(self.features(points,list(history)),device=self.device)).sigmoid().cpu().tolist()
        history.append(points)
        return tracker(points,scores)


def load_tracker(profile=None,checkpoint=None,device='auto'):
    profile=Path(profile or DEFAULT_PROFILE).resolve()
    config=json.loads(profile.read_text(encoding='utf-8-sig'))
    if config.get('format')!='b02-point-profile-v1':raise ValueError('不支持的点模型配置格式')
    if config.get('variant') not in ('control','combined'):raise ValueError('当前发布配置只支持 control 或 combined')
    rule=config['output_rule']
    if (any(not isinstance(rule[k],(int,float)) or isinstance(rule[k],bool) or not isfinite(rule[k]) for k in ('enter','maintain'))
            or not 0<=rule['maintain']<=rule['enter']<=1 or rule.get('grace')!=3):
        raise ValueError('输出门限需要满足 0 <= maintain <= enter <= 1，grace 为 3')
    if config.get('association')!={'score_weight':.5,'evidence_penalty':.15,'strong_observations':6}:
        raise ValueError('该配置的关联规则与当前固定实现不一致')
    checkpoint=Path(checkpoint).resolve() if checkpoint else (profile.parent/config['checkpoint']).resolve()
    if sha256(checkpoint)!=config['checkpoint_sha256']:raise ValueError('推理权重校验失败，请使用与配置配套的文件')
    try:
        import torch
        from experiments.point_scorer.model import PointScorer
        from experiments.point_scorer_expanded.policy import Policy
    except ImportError as error:
        raise ValueError('点模型需要依赖，请先运行 python -m pip install -r requirements-point.txt') from error
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('当前 PyTorch 环境没有可用 CUDA，请用 --device cpu 或配置 GPU 环境')
    if device not in ('cpu','cuda'):raise ValueError('device 需要为 auto、cpu 或 cuda')
    torch.set_num_threads(4)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    metadata=saved.get('metadata',{})
    if metadata.get('format')!='b02-point-inference-v1' or metadata.get('seed')!=config['seed'] or metadata.get('step')!=config['step']:
        raise ValueError('推理权重的类型、训练步数或种子与配置不一致')
    model=PointScorer().to(device)
    model.load_state_dict(saved['model'],strict=True)
    policy=Policy(config['name'],rule['enter'],rule['maintain'],rule['grace'])
    return Handler(model,config['variant'],policy),{**config,'profile':str(profile),'checkpoint':str(checkpoint),'device':device}
