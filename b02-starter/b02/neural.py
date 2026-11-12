"""Optional Track-MT3 loading; baseline users need no Torch dependencies."""
from pathlib import Path
import sys


def load_tracker(checkpoint, upstream, device='auto'):
    root=upstream.resolve()
    pin=root/'UPSTREAM_COMMIT'
    if not pin.is_file() or pin.read_text().strip()!='22c3a19c15df6c560eaadc74462b9b632ede2602':
        raise ValueError('需要由 stage2/prepare_upstream.py 创建的隔离上游目录')
    sys.path.insert(0,str(root))
    try:
        import torch
        from experiments.track_mt3_stage2.adapter import make_model, Handler, HISTORY, MAX_POINTS, QUERIES
    except ImportError as exc:
        raise ValueError('Track-MT3 需要云端实验环境的 PyTorch、NumPy、SciPy、PyYAML 依赖') from exc
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    manifest=saved.get('manifest',{})
    if manifest.get('stage')!='b02-adaptation-pilot-v1':
        raise ValueError('需要 B02 第二阶段权重，不能直接使用二维模拟权重')
    for key,value in [('history_frames',HISTORY),('point_cap_per_frame',MAX_POINTS),('queries',QUERIES)]:
        if manifest.get(key)!=value:
            raise ValueError(f'权重的 {key} 与适配代码不一致，请使用对应版本')
    if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    model=make_model(root,device)
    model.load_state_dict(saved['model'],strict=True)
    model.eval()
    return Handler(model), {**manifest,'checkpoint':str(checkpoint.resolve()),'device':device}
