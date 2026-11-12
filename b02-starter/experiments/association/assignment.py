"""Sparse one-to-one association over the baseline motion gate."""
from heapq import nsmallest
from itertools import product
from math import floor

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def assign(track_ids, tracks, scan, positions, grid, occupied, config, *, scores, score_weight):
    rows, columns, costs = [], [], []
    point_columns = {}
    pending = []
    for batch in track_ids:
        track = tracks[batch]
        elapsed = scan - track.last_scan
        predicted = tuple(x + v * elapsed for x, v in zip(track.position, track.velocity))
        radius = min(1 + 0.2 * (elapsed - 1), 1.6)
        radius_squared = radius * radius
        neighborhood = [range(floor(x - radius), floor(x + radius) + 1) for x in predicted]
        edges = []
        for cell in product(*neighborhood):
            for index in grid.get(cell, ()):
                if index in occupied:
                    continue
                distance = sum((a-b)**2 for a,b in zip(predicted, positions[index]))
                if distance <= radius_squared:
                    cost = distance / radius_squared + score_weight * (1 - scores[index])
                    if cost < 1.15:
                        edges.append((cost, index))
        if not edges:
            continue
        row = len(pending)
        pending.append(batch)
        for cost, index in nsmallest(config.candidates_per_track, edges):
            rows.append(row)
            columns.append(point_columns.setdefault(index, len(point_columns)))
            costs.append(cost + 1e-8)
    if not pending:
        return {}
    point_count = len(point_columns)
    count = len(pending)
    rows.extend(range(count))
    columns.extend(range(point_count, point_count + count))
    costs.extend([1.15 + 1e-8] * count)
    matrix = coo_matrix((costs, (rows, columns)), shape=(count, point_count + count)).tocsr()
    row_ids, column_ids = min_weight_full_bipartite_matching(matrix)
    point_indices = list(point_columns)
    matches = {pending[row]: point_indices[col] for row, col in zip(row_ids, column_ids)
               if col < point_count}
    occupied.update(matches.values())
    return matches
