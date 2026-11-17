"""Frozen 24/8 two-seed independent-vs-one-hop joint association study."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.competition_learning.metrics import evaluate
from experiments.competition_learning.pilot import checkpoint
from experiments.competition_learning.reject_calibration import row_state,rates,fit_offset
from experiments.competition_learning.samples import load
from experiments.graph_association.data import Tables
from experiments.graph_association.model import Association
from experiments.id_recovery.affinity import load_affinity
from experiments.set_association.train import row_losses
from experiments.v2_association.run_stage1 import save,sha


def source_map():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


@torch.inference_mode()
def predict(model,tables):
    d = tables.data
    pair,null = model.graph(d['pair'],d['row'],d['edge_row'],d['edge_col'])
    return pair.cpu().numpy(),null.cpu().numpy()


def checks(control,candidate,rule):
    if control['validation_calibrated'] is None or candidate['validation_calibrated'] is None:
        return {'calibration_feasible':False}
    a,b = control['validation_calibrated'],candidate['validation_calibrated']
    def wrong(x):return x['absent_false_accept_rate']*x['absent_rows']+x['wrong_target_on_present_rows']
    return {'calibration_feasible':True,
        'column_wrong':candidate['selected']['fixed_coverage_column']['wrong'] <=
            control['selected']['fixed_coverage_column']['wrong']*rule['column_wrong_ratio_max'],
        'row_recall':b['positive_match_recall'] >= a['positive_match_recall']-rule['recall_drop_max'],
        'absent_accept':b['absent_false_accept_rate'] <= a['absent_false_accept_rate']*rule['false_accept_ratio_max'],
        'present_wrong':b['wrong_target_on_present_rows'] <= a['wrong_target_on_present_rows']*rule['present_wrong_ratio_max'],
        'total_wrong':wrong(b) <= wrong(a)*rule['total_wrong_ratio_max']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('samples','sample_protocol','prior_models','protocol','prior','affinity','out'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    old_protocol = json.loads(args.sample_protocol.read_text(encoding='utf-8'))
    assert sha(args.sample_protocol) == protocol['sample_protocol_sha256']
    assert source_map() == protocol['new_sources']
    for name,digest in protocol['reused_sources'].items():assert sha(ROOT/name) == digest,name
    trained = json.loads((args.prior_models/'report.json').read_text(encoding='utf-8'))
    assert sha(args.prior_models/'report.json') == protocol['prior_report_sha256']
    assert protocol['training24'] == old_protocol['training24'] and protocol['validation8'] == old_protocol['validation8']
    fit,_ = load(args.samples,protocol['training24'],args.sample_protocol)
    val,_ = load(args.samples,protocol['validation8'],args.sample_protocol)
    assert sha(args.prior) == old_protocol['artifacts']['prior'] and sha(args.affinity) == old_protocol['artifacts']['affinity']
    old = load_affinity(args.affinity,args.prior)
    reference = np.concatenate([old.score(x[:,:93]) for x in np.array_split(val['pair'],40)])
    assert torch.cuda.is_available()
    device = 'cuda'
    ft,vt = Tables(fit,device),Tables(val,device)
    args.out.mkdir(parents=True,exist_ok=False)
    reports,prior_reports = {},{}
    began = monotonic()
    for seed in protocol['seeds']:
        prior = checkpoint(args.prior_models/f'binary-{seed}/selected.pt','binary',args.sample_protocol,trained,seed).to(device)
        with torch.inference_mode():
            base = prior.edges(ft.data['pair']).detach()
            val_base = prior.edges(vt.data['pair']).cpu().numpy()
        fit_state = row_state(fit,base.cpu().numpy(),np.zeros(len(fit['row']),np.float32))
        val_state = row_state(val,val_base,np.zeros(len(val['row']),np.float32))
        prior_reports[str(seed)] = {'validation':rates(val_state,0.),
            'edges':evaluate(val,val_base,np.zeros(len(val['row']),np.float32),reference)}
        for context in (False,True):
            torch.manual_seed(seed);np.random.seed(seed)
            kind = 'context' if context else 'independent'
            model = Association(deepcopy(prior),context).to(device)
            model.initialize(ft.data['row'])
            optimizer = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=.001,weight_decay=.001)
            torch.manual_seed(seed+1)
            folder = args.out/f'{kind}-{seed}';folder.mkdir()
            started = monotonic();history,best,selected = [],float('inf'),None
            for step in range(protocol['steps']+1):
                if monotonic()-started > protocol['budget']['seconds_per_model']:
                    raise TimeoutError('Fixed per-model budget exhausted')
                if step in protocol['checkpoints']:
                    model.eval();pair,null = predict(model,vt)
                    metrics = evaluate(val,pair,null,reference)
                    balanced = row_losses(val,pair,null)['balanced_row_loss']
                    criterion = metrics['binary_loss']+.5*balanced+.5*metrics['column_loss']
                    record = dict(step=step,seconds=monotonic()-started,criterion=criterion,balanced_row_loss=balanced,**metrics)
                    history.append(record)
                    if criterion < best:
                        best,selected = criterion,record
                        torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                            'seed':seed,'kind':kind,'step':step,'protocol_sha256':sha(args.protocol),
                            'sources':source_map()},folder/'selected.pt')
                    save(folder/'history.json',history)
                    print(json.dumps({'kind':kind,'seed':seed,**record}),flush=True)
                if step == protocol['steps']:break
                model.train();loss = ft.loss(model,ft.batch(),base)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],5.);optimizer.step()
            saved = torch.load(folder/'selected.pt',map_location=device,weights_only=True)
            model.load_state_dict(saved['model'],strict=True);model.eval()
            fit_pair,fit_null = predict(model,ft);val_pair,val_null = predict(model,vt)
            fit_candidate = row_state(fit,fit_pair,fit_null);val_candidate = row_state(val,val_pair,val_null)
            offset = fit_offset(fit_state,fit_candidate)
            report = {'selected':selected,'history':history,'offset_training_only':offset,
                'training_raw':rates(fit_candidate,0.),'validation_raw':rates(val_candidate,0.),
                'training_calibrated':rates(fit_candidate,offset) if offset is not None else None,
                'validation_calibrated':rates(val_candidate,offset) if offset is not None else None,
                'weights_sha256':sha(folder/'selected.pt'),'elapsed_seconds':monotonic()-started,
                'trainable_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad)}
            save(folder/'report.json',report);reports[f'{kind}-{seed}'] = report
            del model,optimizer
            torch.cuda.empty_cache()
        del prior,base
        torch.cuda.empty_cache()
    comparisons = {str(seed):checks(reports[f'independent-{seed}'],reports[f'context-{seed}'],protocol['offline_gate'])
        for seed in protocol['seeds']}
    result = {'complete':True,'models':reports,'prior':prior_reports,'comparisons':comparisons,
        'offline_passed':all(all(x.values()) for x in comparisons.values()),'sources':source_map(),
        'protocol_sha256':sha(args.protocol),'elapsed_seconds':monotonic()-began,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
        'data_role':'same parent fit64 head development24/8; repeatedly exposed; not independent whole-system validation'}
    save(args.out/'report.json',result)
    print(json.dumps({'offline_passed':result['offline_passed'],'comparisons':comparisons}),flush=True)


if __name__ == '__main__':main()
