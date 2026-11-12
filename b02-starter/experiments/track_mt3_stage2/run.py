"""Bounded B02 pilot: train-only fitting, causal dev replay, baseline comparison."""
import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--pretrained', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seconds', type=int, default=1200)
    p.add_argument('--evaluate-only', type=Path)
    a = p.parse_args()
    sys.path[:0] = [str(a.upstream.resolve()), str(a.project.resolve())]
    import numpy as np
    import torch
    from adapter import (make_model, transfer, Handler, IgnoreUnknownCriterion,
                         make_window, select_points, HISTORY, MAX_POINTS, QUERIES)
    from b02.replay import _scans, open_scenes, replay
    from b02.algorithm import process_circle, Config
    from b02.scorecard import scorecard_for_run, write_scorecard
    from b02.visualize import export_viewer
    torch.set_num_threads(4)
    torch.manual_seed(20261111); np.random.seed(20261111); random.seed(20261111)
    a.out.mkdir(parents=True, exist_ok=False)
    snapshot=a.out/'source'
    snapshot.mkdir()
    for file in Path(__file__).parent.glob('*.py'):
        shutil.copy2(file,snapshot/file.name)
    shutil.copy2(a.upstream/'b02-stage2.patch',snapshot/'b02-stage2.patch')
    split = json.loads((a.project/'splits/v1.json').read_text())
    train_names, dev_names = split['train'][:4], split['dev'][:2]
    assert not set(train_names) & set(dev_names)
    assert not (set(train_names) | set(dev_names)) & set(split['holdout'])
    manifest = {'stage':'b02-adaptation-pilot-v1', 'train':train_names, 'dev':dev_names,
                'holdout_used':False, 'seed':20261111, 'max_steps':a.steps,
                'training_budget_seconds':a.seconds, 'history_frames':HISTORY,
                'point_cap_per_frame':MAX_POINTS, 'queries':QUERIES,
                'input_features':['Azimuth','Distance','Elevation'],
                'unused_features':['Amp','SNR'], 'clip_windows':4, 'batch':1,
                'pretrained':str(a.pretrained), 'learning_rate':.00005,
                'note':'Fixed subset by existing split order, all dev circles including cold start; local estimated scores only.'}
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    model = make_model(a.upstream)
    transfer_report = transfer(model, a.pretrained)
    (a.out/'transfer.json').write_text(json.dumps(transfer_report, indent=2))
    scenes = {}
    profile = {}
    with open_scenes(a.data) as (_, open_scene):
        for name in train_names:
            with open_scene(name) as source:
                frames = list(_scans(source))
            scenes[name] = frames
            truths = sum(sum(label > 0 for label in labels.values()) for _,_,labels in frames)
            selected_truths = sum(sum(labels[p['PntID']] > 0 for p in select_points(points)) for _,points,labels in frames)
            profile[name] = {'circles':len(frames), 'max_points':max(len(points) for _,points,_ in frames),
                             'max_targets':max(sum(label>0 for label in labels.values()) for _,_,labels in frames),
                             'truth_points':truths, 'retained_truth_input_points':selected_truths}
            if profile[name]['max_targets'] > QUERIES:
                raise ValueError('Increase query capacity based on training profile before fitting')
    (a.out/'data-profile.json').write_text(json.dumps(profile, indent=2))
    criterion = IgnoreUnknownCriterion(model.config.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=.00005)
    started=time.monotonic()
    losses=[]; completed=0
    def save(name):
        temporary=a.out/(name+'.tmp')
        torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                    'step':completed, 'manifest':manifest,
                    'torch_rng':torch.get_rng_state(), 'python_rng':random.getstate()}, temporary)
        temporary.replace(a.out/name)
    if a.evaluate_only:
        model.load_state_dict(torch.load(a.evaluate_only,map_location='cpu')['model'], strict=True)
    else:
        for step in range(1,a.steps+1):
            name = random.choice(train_names)
            frames = scenes[name]
            # Half the clips exercise file cold start; other clips span the scene.
            start = random.randrange(0,min(30,len(frames)-3)) if random.random()<.25 else random.randrange(len(frames)-3)
            history=deque(maxlen=HISTORY); ages={}; clip=[]
            for index in range(max(0,start-HISTORY+1),start+4):
                circle,points,labels=frames[index]
                for label in labels.values():
                    if label>0: ages[label]=ages.get(label,0)+1
                history.append((circle,points,labels))
                if index>=start: clip.append(make_window(history,circle,labels,ages))
            model.train(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            before=time.monotonic()
            total,*components=model([clip],criterion=criterion,teacher_forcing_probability=.5)
            if not torch.isfinite(total): raise FloatingPointError('Nonfinite B02 loss')
            total.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10., error_if_nonfinite=True)
            optimizer.step(); completed=step; losses.append(float(total.detach()))
            event={'event':'train','step':step,'scene':name,'start_circle':start,
                   'loss':losses[-1], 'loss_ma20':sum(losses[-20:])/len(losses[-20:]),
                   'gradient_norm':float(norm), 'seconds':time.monotonic()-before,
                   'elapsed':time.monotonic()-started,
                   'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
            with (a.out/'training.jsonl').open('a') as file: file.write(json.dumps(event)+'\n')
            if step==1 or step%10==0: print(json.dumps(event),flush=True)
            if step%50==0: save('latest.pt')
            if time.monotonic()-started>=a.seconds: break
        save('final.pt')
        # Verify persistence on a causal window without ground truth in model input.
        model.eval()
        clean=make_window([(0,scenes[train_names[0]][0][1],None)],0)
        with torch.no_grad():
            original=model.forward_window(clean)
            restored=make_model(a.upstream)
            restored.load_state_dict(torch.load(a.out/'final.pt',map_location='cpu')['model'],strict=True)
            restored.eval()
            recovered=restored.forward_window(clean)
            torch.testing.assert_close(original.positions,recovered.positions,rtol=0,atol=0)
            torch.testing.assert_close(original.logits,recovered.logits,rtol=0,atol=0)
            del restored
    torch.cuda.empty_cache()
    model.eval()
    print(json.dumps({'event':'evaluation_start','steps':completed,'scenes':dev_names}),flush=True)
    replay(a.data,a.out/'baseline',process_circle,asdict(Config()),dev_names)
    torch.cuda.reset_peak_memory_stats()
    handler=Handler(model)
    replay(a.data,a.out/'candidate',handler,manifest,dev_names)
    peak=torch.cuda.max_memory_allocated()/2**30
    card=scorecard_for_run(a.out/'candidate',a.out/'baseline')
    write_scorecard(a.out/'candidate',card)
    export_viewer(a.data,a.out/'candidate',dev_names[0],a.out/'candidate'/f'{dev_names[0]}.html',a.out/'baseline')
    report={'steps':completed,'stopped_at_budget':completed<a.steps and not a.evaluate_only,
            'elapsed_seconds':time.monotonic()-started,'peak_eval_allocated_gib':peak,
            'values':card['overall']['values'],'baseline':card['overall']['baseline'],
            'inference_telemetry':handler.telemetry,
            'delta':card['overall']['delta'],'holdout_used':False,
            'result':'pilot_complete_not_full_training'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'event':'complete',**report}),flush=True)


if __name__=='__main__':
    main()
