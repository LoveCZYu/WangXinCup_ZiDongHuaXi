"""Bounded smoke run of upstream Track-MT3; not B02 adaptation or paper replication."""
import argparse
import copy
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from track_mt3.config_merge import load_merged_config
from track_mt3.data import MultiTargetSimulator, build_sliding_windows
from track_mt3.models import TrackMT3
from track_mt3.tracking import OnlineTracker
from track_mt3.training import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--frames', type=int, default=60)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite run: {args.out}')
    if not torch.cuda.is_available():
        raise RuntimeError('Stage 1 expects CUDA; refusing silent CPU fallback')
    torch.set_num_threads(2)
    args.out.mkdir(parents=True)
    config = load_merged_config('configs/paper.yaml', 'configs/training/track_mt3.yaml')
    config.training.device = 'cuda'
    config.training.batch_size = 2
    config.training.clip_windows = 2
    config.training.updates = args.steps
    config.training.wandb_enabled = False
    config.training.evaluate_on_checkpoint = False
    config.training.curriculum_warmup_steps = 0
    config.training.output_dir = str(args.out / 'train')
    # Keep every upstream architecture/loss setting; only shorten the run/workload.
    trainer = Trainer(config)
    initial_parameter = next(trainer.model.parameters()).detach().clone()
    parameter_count = sum(p.numel() for p in trainer.model.parameters())
    print(json.dumps({'event':'start', 'parameters':parameter_count, 'gpu':torch.cuda.get_device_name(0), 'steps':args.steps}), flush=True)
    records = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.steps):
        metrics = trainer.train_step()
        if not all(math.isfinite(metrics[k]) for k in ('loss','gradient_norm')):
            raise FloatingPointError('Non-finite loss or gradient')
        records.append(metrics)
        if metrics['step'] == 1 or metrics['step'] % 5 == 0:
            print(json.dumps({k:metrics[k] for k in ('step','loss','gradient_norm','gpu_peak_memory_gb','step_seconds')}), flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    parameter_changed = not torch.equal(initial_parameter, next(trainer.model.parameters()))
    if not parameter_changed:
        raise AssertionError('Parameters did not update')
    checkpoint = trainer.checkpoint('stage1.pt')
    restore_config = copy.deepcopy(config)
    restore_config.training.output_dir = str(args.out / 'restore')
    restored = Trainer(restore_config)
    restored.restore(checkpoint)
    assert restored.state.step == args.steps
    for key, tensor in trainer.model.state_dict().items():
        torch.testing.assert_close(tensor, restored.model.state_dict()[key], rtol=0, atol=0)
    before_resume = next(restored.model.parameters()).detach().clone()
    resume_metrics = restored.train_step()
    assert resume_metrics['step'] == args.steps + 1
    assert math.isfinite(resume_metrics['loss'])
    assert not torch.equal(before_resume, next(restored.model.parameters()))
    final_checkpoint = restored.checkpoint('resumed.pt')
    print(json.dumps({'event':'restore_and_resume_passed','step':resume_metrics['step']}), flush=True)
    del trainer, restored, initial_parameter, before_resume, tensor
    gc.collect()
    torch.cuda.empty_cache()

    # Fresh model: inference GPU memory excludes training optimizer/gradients.
    saved = torch.load(final_checkpoint, map_location='cpu', weights_only=False)
    model = TrackMT3(config)
    model.load_state_dict(saved['model'])
    del saved
    model = model.to('cuda').eval()
    tracker = OnlineTracker(model)
    frames = MultiTargetSimulator(config.simulation, 20261111).simulate(args.frames)
    windows = build_sliding_windows(frames, config.model.window_size, include_partial=True)
    torch.cuda.reset_peak_memory_stats()
    latency = []
    output_counts = []
    with (args.out / 'inference.jsonl').open('w') as handle:
        for frame, window in zip(frames, windows, strict=True):
            torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = tracker.step(window.to('cuda'))
            torch.cuda.synchronize()
            latency.append((time.perf_counter() - start) * 1000)
            assert torch.isfinite(prediction.positions).all()
            assert torch.isfinite(prediction.probabilities).all()
            assert len(set(prediction.track_ids.tolist())) == len(prediction.track_ids)
            output_counts.append(len(prediction.track_ids))
            handle.write(json.dumps({'frame':frame.index,'measurements':frame.measurements.tolist(),
                'true_positions':frame.states[:,:2].tolist(),'true_ids':frame.target_ids.tolist(),
                'positions':prediction.positions.tolist(),'ids':prediction.track_ids.tolist(),
                'probabilities':prediction.probabilities.tolist(),'latency_ms':latency[-1]})+'\n')
    assert len(latency) == args.frames
    timing = latency[5:]
    report = {
        'purpose':'Original 2D simulation smoke test; not B02 validation or paper-score reproduction',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'environment':{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
                       'numpy':np.__version__,'python':__import__('sys').version},
        'model_parameters':parameter_count,'architecture_unchanged':True,
        'training':{'updates':args.steps,'batch_size':2,'clip_windows':2,'curriculum_warmup_steps':0,
                    'seconds':train_seconds,'first_loss':records[0]['loss'],'last_loss':records[-1]['loss'],
                    'all_losses_and_gradients_finite':True,'parameters_updated':parameter_changed,
                    'peak_allocated_gib':max(m['gpu_peak_memory_gb'] for m in records)},
        'checkpoint':{'path':str(final_checkpoint),'exact_parameter_restore':True,'resume_step':resume_metrics['step']},
        'inference':{'frames':len(latency),'includes_partial_windows':True,'mean_ms_excluding_first5':float(np.mean(timing)),
                     'p95_ms_excluding_first5':float(np.percentile(timing,95)),
                     'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                     'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                     'output_count_total':sum(output_counts),'input_points_per_frame_max':max(len(f.measurements) for f in frames)},
    }
    (args.out / 'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (args.out / 'pip-freeze.txt').write_text(subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
