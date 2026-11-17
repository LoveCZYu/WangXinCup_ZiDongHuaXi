"""Truth-assisted references on a frozen output point set, never inference."""
from collections import Counter,defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def shadow_identity(summary):
    """One-to-one whole-scene ID ownership, explicitly not official scoring."""
    targets=summary['per_target']
    batches=sorted({str(batch) for target in targets for batch in target['output_ids']})
    lookup={batch:i for i,batch in enumerate(batches)}
    counts=np.zeros((len(batches),len(targets)),dtype=np.int64)
    for col,target in enumerate(targets):
        for batch,n in target['output_ids'].items():counts[lookup[str(batch)],col]=n
    correct=0
    if counts.size:
        rows,cols=linear_sum_assignment(-counts)
        correct=int(counts[rows,cols].sum())
    minority=sum(int(row.sum()-row.max()) for row in counts) if len(targets) else 0
    return {'truth_points':summary['truth_points'],'matched_points':summary['matched_target_points'],
            'one_to_one_correct_points':correct,'mixed_minority_points':minority,
            'label':'Whole-scene one-output-ID/one-truth-ID diagnostic; not the unknown official matcher'}


def perfect_identity(outputs,frames,clean=False):
    """Retrospective truth IDs; preserve all points unless known-clutter cleaning."""
    allocator={};next_id=1;repaired=[];collisions=0
    for prediction,(_,_,labels) in zip(outputs,frames,strict=True):
        used=Counter();current=[]
        for point in prediction:
            truth=labels[point['PntID']]
            if clean and truth==0:continue
            if truth>0:
                slot=used[truth];used[truth]+=1
                collisions+=slot>0
                key=('truth',truth,slot)
            else:key=('unscored',point['BatchID'])
            if key not in allocator:allocator[key]=next_id;next_id+=1
            current.append({**point,'BatchID':allocator[key]})
        assert len({p['BatchID'] for p in current})==len(current)
        repaired.append(current)
    return repaired,{'same_truth_same_frame_overflows':collisions,'clean_known_clutter':clean,
                     'uses_truth_and_future_identity':'diagnostic only; not deployable'}


def clean_clutter(outputs,frames):
    return [[p for p in prediction if labels[p['PntID']]!=0] for prediction,(_,_,labels) in zip(outputs,frames,strict=True)]


def graph_repair(outputs,frames,graphs):
    """Maximise retrospective main-ID assignments allowed by frozen point edges.

    This is conditional on the actual state trajectory and chosen main IDs, not
    an upper bound on algorithms that would change their own earlier states.
    """
    counts=defaultdict(Counter)
    for prediction,(_,_,labels) in zip(outputs,frames,strict=True):
        for p in prediction:
            truth=labels[p['PntID']]
            if truth>0:counts[truth][p['BatchID']]+=1
    main={truth:max(values,key=values.get) for truth,values in counts.items()}
    repaired=[];changed=eligible=blocked=0
    for prediction,(_,_,labels),edges in zip(outputs,frames,graphs,strict=True):
        if not prediction:repaired.append([]);continue
        options=[];desired=[]
        for p,allowed in zip(prediction,edges,strict=True):
            truth=labels[p['PntID']];goal=main.get(truth,p['BatchID'])
            choices={p['BatchID']}
            if truth>0 and goal in allowed:choices.add(goal)
            eligible+=goal!=p['BatchID'] and goal in choices
            options.append(choices);desired.append(goal)
        ids=sorted(set().union(*options));columns={batch:i for i,batch in enumerate(ids)}
        costs=np.full((len(prediction),len(ids)),1e6)
        for row,(p,choices,goal) in enumerate(zip(prediction,options,desired,strict=True)):
            for batch in choices:
                costs[row,columns[batch]]=-float(batch==goal)-.0001*float(batch==p['BatchID'])
        rows,cols=linear_sum_assignment(costs)
        selected={row:ids[col] for row,col in zip(rows,cols,strict=True)}
        assert all(costs[row,col]<1e5 for row,col in zip(rows,cols,strict=True))
        current=[]
        for row,p in enumerate(prediction):
            batch=selected[row]
            changed+=batch!=p['BatchID']
            blocked+=desired[row]!=p['BatchID'] and desired[row] in options[row] and batch!=desired[row]
            current.append({**p,'BatchID':batch})
        assert len({p['BatchID'] for p in current})==len(current)
        repaired.append(current)
    return repaired,{'eligible_reassignments':eligible,'changed_points':changed,'collision_blocks':blocked,
                     'limitation':'Frozen output points, actual state prefix, retrospective per-truth main IDs; not a global optimum'}
