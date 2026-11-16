"""Freeze one candidate before any new confirmation CSV is opened."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import save,sha


def feasibility(candidate,baseline,limits):
    m,b=candidate['metrics'],baseline['metrics'];c,d=candidate['identity'],baseline['identity']
    return {'point_recall':m['point_recall']>=b['point_recall']-limits['point_recall_drop_max_pp'],
        'track_recall':m['track_recall']>=b['track_recall']-limits['track_recall_drop_max_pp'],
        'clutter':m['clutter_outputs']<=b['clutter_outputs']*limits['clutter_ratio_max'],
        'timeliness':m['timeliness_score']>=b['timeliness_score']-limits['timeliness_drop_max'],
        'purity':c['mixed_minority_truth_points']<=d['mixed_minority_truth_points']*limits['mixed_minority_ratio_max'],
        'weighted_total':m['weighted_total']>=b['weighted_total']+limits['weighted_gain_min'],
        'fragmentation':m['fragmentation_score']>=b['fragmentation_score'],
        'continuous':m['continuous_score']>=b['continuous_score']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',nargs=2,type=Path,required=True);p.add_argument('--checkpoints',nargs=2,type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--identity',choices=('physical','geo4','learned4','learned8'),default='physical')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--require-goal',action='store_true');a=p.parse_args()
    from experiments.v3_tracking.runtime import load_scorer
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'));baseline=json.loads(a.baseline.read_text(encoding='utf-8'))
    assert baseline['complete'] and baseline['scenes']==plan['calibration']
    assert not a.out.exists();models=[];checks={};goals={}
    for seed,path,checkpoint in zip(plan['seeds'],a.reports,a.checkpoints):
        report=json.loads(path.read_text(encoding='utf-8'))
        assert report['complete'] and report['scenes']==plan['calibration'] and not report['holdout_used']
        evidence=json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
        assert evidence['checkpoint_sha256']==sha(checkpoint)
        if 'results' in report:
            assert report['point_sets_identical']
            result=report['results'][a.identity]
        else:
            assert a.identity=='physical';result=report
        scorer,memory,meta,step=load_scorer(checkpoint)
        assert meta['seed']==seed
        selected=json.loads((checkpoint.parent/'selected.json').read_text(encoding='utf-8'))
        assert selected['selected']['step']==step
        if 'results' in report and meta['format']=='b02-residual-association-v1':assert all(report['physical_reference_equal'].values())
        fit=meta['pairs']['fit'];assert not set(fit)&set(plan['confirmation16']+plan['calibration']+plan['development_regression']+plan['sealed_excluded'])
        checks[str(seed)]=feasibility(result,baseline,plan['calibration_feasible'])
        m,b=result['metrics'],baseline['metrics'];c,d=result['identity'],baseline['identity'];goal=plan['identity_goal']
        goals[str(seed)]={'fragmentation':m['fragmentation_score']>=b['fragmentation_score']+goal['fragmentation_gain_min'],
            'continuous':m['continuous_score']>=b['continuous_score']+goal['continuous_gain_min'],
            'switches':c.get('switches',0)<=d.get('switches',0)*goal['switch_ratio_max']}
        if 'results' in report:
            checks[str(seed)]['upstream_identity_purity']=c['mixed_minority_truth_points']<=report['results']['physical']['identity']['mixed_minority_truth_points']*1.05
        models.append({'seed':seed,'step':step,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'memory':memory,
            'calibration_report':str(path),'calibration_report_sha256':sha(path),'calibration_metrics':m,'calibration_identity':c})
    passed=all(all(v.values()) for v in checks.values());goal_passed=all(all(v.values()) for v in goals.values())
    save(a.out.with_suffix('.checks.json'),{'feasible':passed,'goal_passed':goal_passed,'checks':checks,'goals':goals})
    if not passed or (a.require_goal and not goal_passed):raise ValueError('Candidate not eligible; confirmation remains closed')
    files=[p for p in Path(__file__).parent.glob('*.py')]
    freeze={'format':'b02-v3-confirmation-freeze-v1','checks_passed':True,'major_goal_passed':goal_passed,'checks':checks,'goals':goals,
        'models':models,'selected_seed':plan['output_seed'],'identity':a.identity,'strategy':'rank_only','scale':1.,
        'confirmation_scenes':plan['confirmation16'],'confirmation_policy':'fixed one candidate; no selection or tuning from confirmation',
        'point_weights_sha256':sha(ROOT/'assets/point-scorer-8000.pt'),'point_profile_sha256':sha(ROOT/'profiles/current.json'),
        'baseline_profile_sha256':sha(ROOT/'profiles/v2-association-candidate.json'),'baseline_head_sha256':sha(ROOT/'assets/v2-association-candidate.pt'),
        'baseline_calibration_report_sha256':sha(a.baseline),'split_sha256':sha(ROOT/'splits/v3_tracking.json'),
        'protocol_sha256':sha(Path(__file__).parent/'protocol.json'),'runtime_sha256':{p.name:sha(p) for p in files},
        'confirmation_used':False,'holdout_used':False,'official_submission':False}
    save(a.out,freeze);print(json.dumps({'frozen':str(a.out),'goal_passed':goal_passed,'checks':checks}),flush=True)


if __name__=='__main__':main()
