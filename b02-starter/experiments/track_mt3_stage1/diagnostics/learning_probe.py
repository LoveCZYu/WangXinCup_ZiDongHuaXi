"""Bounded learning probes, preserving upstream model and loss settings."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from track_mt3.config import ExperimentConfig, _construct
from track_mt3.training import Trainer, generate_training_clips
from track_mt3.tracking import OnlineTracker
from track_mt3.metrics import gospa
from diagnose_model import evaluate


@torch.no_grad()
def fixed_evaluate(model, clips):
    model.eval()
    records=[]
    for clip in clips:
        tracker=OnlineTracker(model)
        for w in clip:
            out=tracker.step(w.to('cuda'))
            result=gospa(out.positions.numpy(),w.target_positions.numpy())
            pr=out.prediction
            probs=pr.probabilities.flatten().cpu().numpy()
            records.append({'targets':len(w.target_ids),'outputs':len(out.track_ids),
                            'max_probability':float(probs.max()),'gospa':result.value,
                            'matched':len(result.matched_pairs)})
    return {'mean_'+k:float(np.mean([r[k] for r in records])) for k in records[0]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['fixed','random'],required=True)
    p.add_argument('--steps',type=int,required=True)
    p.add_argument('--teacher-forcing-zero',action='store_true')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    torch.set_num_threads(2)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    config=_construct(ExperimentConfig,saved['config'])
    config.training.output_dir=str(a.out/'train')
    config.training.wandb_enabled=False
    if a.teacher_forcing_zero:
        config.training.teacher_forcing_start=0.0
        config.training.teacher_forcing_end=0.0
    trainer=Trainer(config)
    trainer.restore(a.checkpoint)
    del saved
    fixed_clips=generate_training_clips(config,np.random.SeedSequence(90119)) if a.mode=='fixed' else None
    evaluations=[]
    def assess():
        if fixed_clips is not None:
            summary=fixed_evaluate(trainer.model,fixed_clips)
        else:
            summary,_=evaluate(trainer.model,config,seeds=(20261111,))
        record={'step':trainer.state.step,**summary}
        evaluations.append(record)
        print(json.dumps({'event':'evaluate','mode':a.mode,**record}),flush=True)
    assess()
    start=time.perf_counter()
    loss=[]
    for i in range(a.steps):
        m=trainer.train_step(fixed_clips)
        assert math.isfinite(m['loss']) and math.isfinite(m['gradient_norm'])
        loss.append(m['loss'])
        if (i+1)%50==0:
            print(json.dumps({'event':'train','mode':a.mode,'step':m['step'],'loss_ma50':float(np.mean(loss[-50:]))}),flush=True)
        if (i+1)% (100 if a.mode=='fixed' else 500)==0:
            assess()
    if evaluations[-1]['step']!=trainer.state.step: assess()
    checkpoint=trainer.checkpoint('probe.pt')
    report={'mode':a.mode,'teacher_forcing_zero':a.teacher_forcing_zero,'initial_checkpoint':a.checkpoint,'added_updates':a.steps,
            'checkpoint':str(checkpoint),'seconds':time.perf_counter()-start,
            'loss_first20_mean':float(np.mean(loss[:20])),'loss_last20_mean':float(np.mean(loss[-20:])),
            'evaluations':evaluations}
    (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'event':'complete',**report}),flush=True)


if __name__=='__main__':main()
