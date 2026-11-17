"""Fixed-coverage identity and row-rejection metrics on head validation only."""
import numpy as np


def logsumexp(logits, mask):
    return np.logaddexp.reduce(np.where(mask, logits, -1e4), axis=1)


def evaluate(data, pair, null, reference_pair=None):
    known = data['edge_known']
    y = data['edge_positive']
    binary = float((np.logaddexp(0., pair[known])-y[known]*pair[known]).mean())
    ri, ci = data['row_index'], data['col_index']
    rv, cv = data['row_mask'], data['col_mask']
    rp = data['edge_positive'][ri] & rv
    cp = data['col_target'][ci] & cv
    row = np.concatenate((pair[ri], null[:, None]), 1)
    row_valid = np.concatenate((rv, np.ones((len(rv), 1), dtype=bool)), 1)
    row_target = np.concatenate((rp, ~rp.any(1, keepdims=True)), 1)
    r = data['row_allowed']
    row_loss = float((logsumexp(row, row_valid)-logsumexp(row, row_target))[r].mean())
    prediction = np.where(row_valid, row, -1e4).argmax(1)
    correct = row_target[np.arange(len(row)), prediction]
    positive_rows = r & rp.any(1)
    absent_rows = r & ~rp.any(1)
    assert positive_rows.any() and absent_rows.any(), 'Both continuation and rejection cases are required'
    positive_recall = float(correct[positive_rows].mean())
    absent_accept = float((prediction[absent_rows] < rv.shape[1]).mean())
    column = np.concatenate((pair[ci], np.zeros((len(ci), 1), dtype=pair.dtype)), 1)
    col_valid = np.concatenate((cv, np.ones((len(cv), 1), dtype=bool)), 1)
    col_target = np.concatenate((cp, ~cp.any(1, keepdims=True)), 1)
    c = data['col_allowed']
    col_loss = float((logsumexp(column, col_valid)-logsumexp(column, col_target))[c].mean())
    result = {'binary_loss': binary, 'row_loss': row_loss, 'column_loss': col_loss,
        'joint_loss': binary+.5*row_loss+.5*col_loss,
        'positive_row_match_recall': positive_recall, 'absent_row_false_accept_rate': absent_accept,
        'positive_rows': int(positive_rows.sum()), 'absent_rows': int(absent_rows.sum())}
    selected_groups = np.flatnonzero(data['col_eval'])
    assert len(selected_groups)
    logits = np.where(cv[selected_groups], pair[ci[selected_groups]], -1e4)
    winner = logits.argmax(1)
    top = logits[np.arange(len(logits)), winner]
    right = cp[selected_groups][np.arange(len(logits)), winner]
    ref = pair if reference_pair is None else reference_pair
    reference = np.where(cv[selected_groups], ref[ci[selected_groups]], -1e4).max(1)
    count = int((reference >= 0.).sum())
    assert count > 0, 'Old affinity has no accepted reference coverage; stop rather than redefine the metric'
    chosen = np.argsort(-top, kind='stable')[:count]
    result['fixed_coverage_column'] = {'groups': len(selected_groups), 'selected': count,
        'correct': int(right[chosen].sum()), 'wrong': count-int(right[chosen].sum()),
        'reference': 'old affinity score>=0; identical selected count for all models',
        'positive_possible_groups': int(cp[selected_groups].any(1).sum())}
    return result
