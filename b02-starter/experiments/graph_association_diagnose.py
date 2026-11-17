"""Post-training frozen-row swaps and joint-target feasibility audit only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.competition_learning.model import Model
from experiments.competition_learning.reject_calibration import row_state,rates
from experiments.competition_learning.samples import load
from experiments.graph_association.data import Tables
from experiments.graph_association.model import Association
from experiments.graph_association.train import predict,source_map
from experiments.set_association.model import geometry
from experiments.v2_association.run_stage1 import save,sha


def audit_targets(data):
    _,_,supervised,positive = geometry(data)
    rows,cols = len(data['row']),len(data['col_allowed'])
    valid_positive = data['edge_positive'] & supervised[data['edge_row']]
    canonical = data['col_target'] & valid_positive
    row_positive = np.bincount(data['edge_row'][valid_positive],minlength=rows)>0
    row_canonical = np.bincount(data['edge_row'][canonical],minlength=rows)>0
    graph = csr_matrix((np.ones(valid_positive.sum(),np.int8),
        (data['edge_row'][valid_positive],data['edge_col'][valid_positive])),shape=(rows,cols))
    matched = maximum_bipartite_matching(graph,perm_type='column')
    capacity = int((matched>=0).sum())
    degree = np.bincount(data['edge_col'][valid_positive],minlength=cols)
    return {'supervised_positive_rows':int(row_positive.sum()),'maximum_simultaneous_correct_rows':capacity,
        'minimum_positive_rows_unsatisfied_by_one_to_one':int(row_positive.sum())-capacity,
        'positive_rows_without_any_preferred_column_edge':int((row_positive & ~row_canonical).sum()),
        'positive_edges':int(valid_positive.sum()),'semantic_positive_not_column_preferred_edges':int((valid_positive & ~canonical).sum()),
        'columns_with_multiple_semantic_positive_rows':int((degree>1).sum()),
        'scope':'Frozen, disjoint recorded assignment stages, reliable anchored rows with all candidate labels known. Not whole-system achievable score.'}


def behavior(data,pair,null):
    state = row_state(data,pair,null)
    accepted = state['margin']>0
    allowed = state['positive'] | state['absent']
    good = accepted & state['correct'] & state['positive']
    bad = accepted & ((~state['correct'] & state['positive']) | state['absent'])
    return state,dict(rates(state,0.),
        correct_top_even_before_rejection=int((state['correct'] & state['positive']).sum()),
        wrong_top_even_before_rejection=int((~state['correct'] & state['positive']).sum()),
        correct_accepted=int(good.sum()),wrong_accepted=int(bad.sum())),good,bad,allowed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('samples','sample_protocol','protocol','training','diagnostic','out'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    plan = json.loads(args.diagnostic.read_text(encoding='utf-8'))
    assert sha(Path(__file__)) == plan['source_sha256']
    assert sha(args.protocol) == plan['training_protocol_sha256']
    assert sha(args.training/'report.json') == plan['training_report_sha256']
    assert plan['validation8'] == protocol['validation8']
    assert sha(args.sample_protocol) == protocol['sample_protocol_sha256']
    report = json.loads((args.training/'report.json').read_text(encoding='utf-8'))
    data,_ = load(args.samples,protocol['validation8'],args.sample_protocol)
    tables = Tables(data,'cuda')
    audit = audit_targets(data)
    seeds = {}
    for seed in protocol['seeds']:
        outputs = {}
        for kind in ('independent','context'):
            record = report['models'][f'{kind}-{seed}']
            path = args.training/f'{kind}-{seed}/selected.pt'
            assert sha(path) == record['weights_sha256']
            saved = torch.load(path,map_location='cpu',weights_only=True)
            assert saved['sources'] == source_map() == protocol['new_sources']
            assert saved['protocol_sha256'] == sha(args.protocol) and saved['seed']==seed and saved['kind']==kind
            model = Association(Model(),kind=='context').cuda().eval()
            model.load_state_dict(saved['model'],strict=True)
            pair,null = predict(model,tables)
            offset = record['offset_training_only']
            assert offset is not None
            outputs[kind] = pair,null+offset
            del model
        variants = {}
        states = {}
        for kind,pair_from,null_from in (
            ('independent','independent','independent'),('context_edges_only','context','independent'),
            ('context_null_only','independent','context'),('context','context','context')):
            state,metrics,good,bad,allowed = behavior(data,outputs[pair_from][0],outputs[null_from][1])
            variants[kind] = metrics
            states[kind] = state,good,bad,allowed
        a,ag,ab,_ = states['independent'];b,bg,bb,_ = states['context']
        positive = a['positive']
        assert np.array_equal(positive,b['positive'])
        ranking = {'wrong_to_correct_top':int((~a['correct'] & b['correct'] & positive).sum()),
            'correct_to_wrong_top':int((a['correct'] & ~b['correct'] & positive).sum()),
            'new_correct_accepts':int((bg & ~ag).sum()),'lost_correct_accepts':int((ag & ~bg).sum()),
            'new_wrong_accepts':int((bb & ~ab).sum()),'removed_wrong_accepts':int((ab & ~bb).sum())}
        prior_correct = int(round(report['prior'][str(seed)]['validation']['positive_match_recall']*int(positive.sum())))
        seeds[str(seed)] = {'variants':variants,'paired_transitions':ranking,
            'prior_local_correct_accepts':prior_correct,
            'prior_local_correct_accepts_above_max_joint_capacity':max(0,prior_correct-audit['maximum_simultaneous_correct_rows'])}
    save(args.out,{'research_only':True,'new_training':False,'threshold_search':False,
        'original_gate_overridden':False,'independent_validation':False,
        'diagnostic_protocol_sha256':sha(args.diagnostic),'target_feasibility':audit,'seeds':seeds,
        'hybrid_caveat':'Swap already trained edge and null functions incl their existing training offsets; no recalibration. Numerical attribution only, not deployment candidates or a causal identification of the underlying data process.'})
    print(json.dumps({'target_feasibility':audit,'seeds':seeds},indent=2),flush=True)


if __name__ == '__main__':main()
