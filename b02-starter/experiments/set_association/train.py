"""Preregistered orthogonal ablations on existing 24/8 cached rollouts."""
import argparse
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.competition_learning.metrics import evaluate
from experiments.competition_learning.model import objective, set_loss
from experiments.competition_learning.pilot import checkpoint
from experiments.competition_learning.reject_calibration import row_state, rates, fit_offset
from experiments.competition_learning.samples import load, Tensors
from experiments.competition_learning.train import predict
from experiments.id_recovery.affinity import load_affinity
from experiments.set_association.model import Identity, Rejection, full_pairs, geometry
from experiments.v2_association.run_stage1 import save, sha


@torch.inference_mode()
def null_predict(model, data, device):
    index, mask, _, _ = geometry(data)
    values = []
    for start in range(0, len(index), 2048):
        sl = slice(start, start+2048)
        values.append(model(torch.as_tensor(data['row'][sl], device=device),
            torch.as_tensor(data['pair'][index[sl]], device=device),
            torch.as_tensor(mask[sl], device=device)).cpu().numpy())
    return np.concatenate(values)


def row_losses(data, pair, null):
    index, mask, supervised, positive = geometry(data)
    logits = np.concatenate((pair[index], null[:, None]), 1)
    valid = np.concatenate((mask, np.ones((len(mask), 1), bool)), 1)
    targets = np.concatenate((data['edge_positive'][index] & mask, ~positive[:, None]), 1)
    def lse(values, ok):return np.logaddexp.reduce(np.where(ok, values, -1e4), axis=1)
    loss = lse(logits, valid)-lse(logits, targets)
    return {'row_loss': float(loss[supervised].mean()),
        'balanced_row_loss': float(.5*(loss[supervised & positive].mean()+loss[supervised & ~positive].mean()))}


def source_map():
    return {str(p.relative_to(ROOT)).replace('\\', '/'):sha(p) for p in
        sorted(Path(__file__).parent.glob('*.py'))}


def write_weights(path, model, step, seed, kind, protocol):
    torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
        'step':step, 'seed':seed, 'kind':kind, 'protocol_sha256':sha(protocol),
        'sources':source_map()}, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('samples', 'sample_protocol', 'prior_models', 'protocol', 'prior', 'affinity', 'out'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    old_protocol = json.loads(args.sample_protocol.read_text(encoding='utf-8'))
    assert sha(args.sample_protocol) == protocol['sample_protocol_sha256']
    assert source_map() == protocol['new_sources']
    for path, digest in protocol['reused_sources'].items():assert sha(ROOT/path) == digest, path
    trained = json.loads((args.prior_models/'report.json').read_text(encoding='utf-8'))
    assert sha(args.prior_models/'report.json') == protocol['prior_report_sha256']
    assert protocol['training24'] == old_protocol['training24'] and protocol['validation8'] == old_protocol['validation8']
    fit, _ = load(args.samples, protocol['training24'], args.sample_protocol)
    val, _ = load(args.samples, protocol['validation8'], args.sample_protocol)
    assert sha(args.prior) == old_protocol['artifacts']['prior'] and sha(args.affinity) == old_protocol['artifacts']['affinity']
    old = load_affinity(args.affinity, args.prior)
    reference = np.concatenate([old.score(x[:, :93]) for x in np.array_split(val['pair'], 40)])
    args.out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    device = 'cuda'
    fit_full = dict(fit, pair=full_pairs(fit['pair'], fit['row'][fit['edge_row']]))
    val_full = dict(val, pair=full_pairs(val['pair'], val['row'][val['edge_row']]))
    identity_reports, rejection_reports, prior_reports = {}, {}, {}
    began = monotonic()
    for seed in protocol['seeds']:
        prior = checkpoint(args.prior_models/f'binary-{seed}/selected.pt', 'binary',
            args.sample_protocol, trained, seed).to(device)
        prior.eval()
        fit_pair, _ = predict(prior, fit, device)
        val_pair, _ = predict(prior, val, device)
        prior_reports[str(seed)] = evaluate(val, val_pair, np.zeros(len(val['row']), np.float32), reference)
        for full in (False, True):
            torch.manual_seed(seed);np.random.seed(seed)
            kind = 'full186' if full else 'continued96'
            data = fit_full if full else fit
            validation = val_full if full else val
            model = Identity(full).to(device)
            model.initialize(prior, torch.as_tensor(data['pair'][data['known_ids']], device=device))
            # Enforce exact retained score before either extra-training arm.
            torch.testing.assert_close(model.edges(torch.as_tensor(data['pair'][:2048], device=device)),
                prior.edges(torch.as_tensor(fit['pair'][:2048], device=device)), atol=2e-5, rtol=2e-5)
            arrays = Tensors(data, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
            # Different input dimensions consume different initialization draws;
            # reset sampling so the two arms see exactly the same row/column IDs.
            torch.manual_seed(seed+1)
            folder = args.out/f'{kind}-{seed}';folder.mkdir()
            history, best, selected = [], float('inf'), None
            started = monotonic()
            for step in range(protocol['identity']['steps']+1):
                assert monotonic()-started < protocol['budget']['seconds_per_model']
                if step in protocol['checkpoints']:
                    model.eval()
                    pair, null = predict(model, validation, device)
                    m = evaluate(validation, pair, null, reference)
                    record = dict(step=step, seconds=monotonic()-started, **m)
                    history.append(record)
                    if m['binary_loss'] < best:
                        best, selected = m['binary_loss'], record
                        write_weights(folder/'selected.pt', model, step, seed, kind, args.protocol)
                    save(folder/'history.json', history)
                    print(json.dumps({'study':'identity','kind':kind,'seed':seed,**record}), flush=True)
                if step == protocol['identity']['steps']:break
                model.train();loss = objective(model, arrays.batch(), False)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.);optimizer.step()
            identity_reports[f'{kind}-{seed}'] = {'selected':selected,'history':history,
                'weights_sha256':sha(folder/'selected.pt'),'elapsed_seconds':monotonic()-started}
            del model, optimizer, arrays
            torch.cuda.empty_cache()
        # Null models always condition on the frozen prior96, never the selected identity arm.
        fi, fm, fs, fp = geometry(fit)
        frow = torch.as_tensor(fit['row'], device=device)
        fedge = torch.as_tensor(fit['pair'], device=device)
        findex = torch.as_tensor(fi, device=device);fmask = torch.as_tensor(fm, device=device)
        frozen_logits = torch.as_tensor(fit_pair, device=device)
        labels = torch.as_tensor(fit['edge_positive'], device=device)
        ids = torch.as_tensor(np.flatnonzero(fs), device=device)
        pos = torch.as_tensor(np.flatnonzero(fs & fp), device=device)
        neg = torch.as_tensor(np.flatnonzero(fs & ~fp), device=device)
        fit_state = row_state(fit, fit_pair, np.zeros(len(fit['row']), np.float32))
        val_state = row_state(val, val_pair, np.zeros(len(val['row']), np.float32))
        for balanced in (False, True):
            for context in (False, True):
                torch.manual_seed(seed);np.random.seed(seed)
                kind = ('candidate' if context else 'history')+('-balanced' if balanced else '-uniform')
                model = Rejection(context).to(device)
                model.initialize(prior, frow)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
                torch.manual_seed(seed+1)
                folder = args.out/f'{kind}-{seed}';folder.mkdir()
                history, best, selected = [], float('inf'), None
                started = monotonic()
                for step in range(protocol['rejection']['steps']+1):
                    assert monotonic()-started < protocol['budget']['seconds_per_model']
                    if step in protocol['checkpoints']:
                        model.eval()
                        null = null_predict(model, val, device)
                        metrics = row_losses(val, val_pair, null)
                        record = dict(step=step,seconds=monotonic()-started,**metrics)
                        history.append(record)
                        if metrics['balanced_row_loss'] < best:
                            best, selected = metrics['balanced_row_loss'], record
                            write_weights(folder/'selected.pt', model, step, seed, kind, args.protocol)
                        save(folder/'history.json', history)
                        print(json.dumps({'study':'rejection','kind':kind,'seed':seed,**record}), flush=True)
                    if step == protocol['rejection']['steps']:break
                    def choose(pool, n):return pool[torch.randint(len(pool), (n,), device=device)]
                    ri = torch.cat((choose(pos, 128),choose(neg,128))) if balanced else choose(ids,256)
                    idx, mask = findex[ri], fmask[ri]
                    null = model(frow[ri],fedge[idx],mask)
                    logits = torch.cat((frozen_logits[idx],null[:, None]),1)
                    target = labels[idx] & mask
                    valid = torch.cat((mask,torch.ones(len(ri),1,device=device,dtype=torch.bool)),1)
                    target = torch.cat((target,~target.any(1,keepdim=True)),1)
                    loss = set_loss(logits,valid,target)
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                saved = torch.load(folder/'selected.pt',map_location=device,weights_only=True)
                model.load_state_dict(saved['model'],strict=True);model.eval()
                fit_null = null_predict(model,fit,device);val_null = null_predict(model,val,device)
                fitted = row_state(fit,fit_pair,fit_null);validated = row_state(val,val_pair,val_null)
                offset = fit_offset(fit_state,fitted)
                assert offset is not None
                report = {'selected':selected,'history':history,'offset_training_only':offset,
                    'training_raw':rates(fitted,0.),'training_calibrated':rates(fitted,offset),
                    'validation_raw':rates(validated,0.),'validation_calibrated':rates(validated,offset),
                    'prior_training':rates(fit_state,0.),'prior_validation':rates(val_state,0.),
                    'weights_sha256':sha(folder/'selected.pt'),'elapsed_seconds':monotonic()-started}
                save(folder/'report.json',report)
                rejection_reports[f'{kind}-{seed}'] = report
                del model,optimizer
                torch.cuda.empty_cache()
        del prior,frow,fedge,findex,fmask,frozen_logits,labels,ids,pos,neg
        torch.cuda.empty_cache()
    identity_checks, rejection_checks = {}, {}
    for seed in protocol['seeds']:
        control = identity_reports[f'continued96-{seed}']['selected']
        full = identity_reports[f'full186-{seed}']['selected']
        rule = protocol['identity']['gate']
        checks = {'wrong_columns':full['fixed_coverage_column']['wrong'] <= control['fixed_coverage_column']['wrong']*rule['wrong_ratio_max'],
            'positive_recall':full['positive_row_match_recall'] >= control['positive_row_match_recall']-rule['recall_drop_max'],
            'absent_accept':full['absent_row_false_accept_rate'] <= control['absent_row_false_accept_rate']*rule['false_accept_ratio_max']}
        identity_checks[str(seed)] = {'checks':checks,'passed':all(checks.values())}
        history = rejection_reports[f'history-balanced-{seed}']['validation_calibrated']
        candidate = rejection_reports[f'candidate-balanced-{seed}']['validation_calibrated']
        rule = protocol['rejection']['gate']
        def wrong(x):return x['absent_false_accept_rate']*x['absent_rows']+x['wrong_target_on_present_rows']
        checks = {'absent_accept':candidate['absent_false_accept_rate'] <= history['absent_false_accept_rate']*rule['false_accept_ratio_max'],
            'positive_recall':candidate['positive_match_recall'] >= history['positive_match_recall']-rule['recall_drop_max'],
            'present_wrong':candidate['wrong_target_on_present_rows'] <= history['wrong_target_on_present_rows']*rule['present_wrong_ratio_max'],
            'total_wrong':wrong(candidate) <= wrong(history)*rule['total_wrong_ratio_max']}
        rejection_checks[str(seed)] = {'checks':checks,'passed':all(checks.values())}
    result = {'complete':True,'protocol_sha256':sha(args.protocol),'prior':prior_reports,
        'identity':identity_reports,'rejection':rejection_reports,'identity_checks':identity_checks,
        'rejection_checks':rejection_checks,'identity_passed':all(x['passed'] for x in identity_checks.values()),
        'rejection_passed':all(x['passed'] for x in rejection_checks.values()),'sources':source_map(),
        'elapsed_seconds':monotonic()-began,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
        'data_role':'head development inside parent fit64; not independent whole-system validation',
        'excluded_unknown_rows':{'fit':row_state(fit,np.zeros(len(fit['pair']),np.float32),np.zeros(len(fit['row']),np.float32))['excluded_unknown_rows'],
            'validation':val_state['excluded_unknown_rows']}}
    save(args.out/'report.json',result)
    print(json.dumps({'identity_checks':identity_checks,'rejection_checks':rejection_checks}),flush=True)


if __name__ == '__main__':main()
