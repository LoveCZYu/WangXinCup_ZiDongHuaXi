"""Scene-paired uncertainty without treating points as independent samples."""
import numpy as np


def sufficient(scene,identity):
    targets=scene['per_target'];long=[t for t in targets if t['truth_points']>2]
    detected=[t for t in targets if t['output_ids']]
    return np.asarray([scene['truth_points'],scene['matched_target_points'],scene['clutter_outputs'],
        len(targets),sum(t['matched_points']/t['truth_points'] for t in targets),
        len(detected),sum(1/len(t['output_ids']) for t in detected),len(long),
        sum(min(1,t['most_matched_points_with_one_id']/(t['truth_points']-2)) for t in long),
        sum(0 if t['first_output_delay_circles'] is None else 1-min(50,max(0,(t['first_output_delay_circles']-2)*5))/50 for t in long),
        identity['mixed_minority_truth_points']],dtype=np.float64)


def metrics(values):
    v=np.asarray(values);truth=np.maximum(v[...,0],1);targets=np.maximum(v[...,3],1)
    point=100*v[...,1]/truth;track=100*v[...,4]/targets
    clutter=100*np.maximum(0,1-v[...,2]/truth)
    fragmentation=100*v[...,6]/np.maximum(v[...,5],1)
    continuous=100*v[...,8]/np.maximum(v[...,7],1)
    timely=100*v[...,9]/np.maximum(v[...,7],1)
    total=.3*track+.1*clutter+.2*fragmentation+.2*continuous+.2*timely
    mixed_rate=100*v[...,10]/np.maximum(v[...,1],1)
    return {'point_recall':point,'track_recall':track,'clutter_score':clutter,'fragmentation_score':fragmentation,
            'continuous_score':continuous,'timeliness_score':timely,'weighted_total':total,'mixed_rate_percent':mixed_rate}


def paired_bootstrap(baseline,candidate,count=10000,seed=20261116):
    baseline=np.asarray(baseline);candidate=np.asarray(candidate)
    assert baseline.shape==candidate.shape and baseline.ndim==2
    rng=np.random.default_rng(seed);draw=rng.integers(0,len(baseline),size=(count,len(baseline)))
    base=metrics(baseline[draw].sum(1));current=metrics(candidate[draw].sum(1))
    result={}
    for name in base:
        delta=current[name]-base[name]
        result[name]={'lower':float(np.quantile(delta,.025)),'upper':float(np.quantile(delta,.975))}
    return result
