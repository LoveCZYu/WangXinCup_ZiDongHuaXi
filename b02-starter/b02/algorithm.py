"""CPU reference tracker. Inputs and outputs use the official point fields."""

from collections import defaultdict
from dataclasses import dataclass
from heapq import nsmallest
from itertools import product
from math import floor


@dataclass(frozen=True)
class Config:
    # Normalized-coordinate gates, with velocity measured per received scan.
    gate: tuple[float, float, float] = (0.025, 0.065, 0.020)
    confirm_hits: int = 3
    max_missed: int = 3
    max_tentative: int = 256
    candidates_per_track: int = 12
    velocity_gain: float = 0.35


DEFAULT_CONFIG = Config()
POSITION = ("Azimuth", "Elevation", "Distance")
OUTPUT_FIELDS = ("Time", *POSITION, "CircleID", "PntID")


@dataclass
class Track:
    position: tuple[float, float, float]
    last_scan: int
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hits: int = 1


def _associate(track_ids, tracks, scan, positions, grid, occupied, config):
    edges = []
    for batch in track_ids:
        track = tracks[batch]
        elapsed = scan - track.last_scan
        predicted = tuple(x + v * elapsed for x, v in zip(track.position, track.velocity))
        radius = min(1 + 0.2 * (elapsed - 1), 1.6)
        neighborhood = [range(floor(x - radius), floor(x + radius) + 1) for x in predicted]

        def candidates():
            for cell in product(*neighborhood):
                for index in grid.get(cell, ()):
                    if index in occupied:
                        continue
                    distance = sum((a - b) ** 2 for a, b in zip(predicted, positions[index]))
                    if distance <= radius**2:
                        yield distance, batch, index

        edges.extend(nsmallest(config.candidates_per_track, candidates()))

    assigned = {}
    for _, batch, index in sorted(edges):
        if batch not in assigned and index not in occupied:
            assigned[batch] = index
            occupied.add(index)
    return assigned


def _output(point, batch):
    return {**{field: point[field] for field in OUTPUT_FIELDS}, "BatchID": batch}


def process_circle(points, state, *, config=DEFAULT_CONFIG, assignment=_associate):
    """Consume one scan, retain state within the packet, and select original points."""
    scan = state.get("scan", 0) + 1
    state["scan"] = scan
    tracks = state.setdefault("tracks", {})
    next_id = state.get("next_id", 1)
    positions = [tuple(p[key] / scale for key, scale in zip(POSITION, config.gate)) for p in points]
    grid = defaultdict(list)
    for i, position in enumerate(positions):
        grid[tuple(map(floor, position))].append(i)

    confirmed = [batch for batch, t in tracks.items() if t.hits >= config.confirm_hits]
    tentative = [batch for batch, t in tracks.items() if t.hits < config.confirm_hits]
    occupied = set()
    matches = assignment(confirmed, tracks, scan, positions, grid, occupied, config)
    matches.update(assignment(tentative, tracks, scan, positions, grid, occupied, config))
    outputs = []
    for batch, track in list(tracks.items()):
        index = matches.get(batch)
        elapsed = scan - track.last_scan
        if index is None:
            if track.hits < config.confirm_hits or elapsed > config.max_missed:
                del tracks[batch]
            continue

        position = positions[index]
        gain = 1.0 if track.hits == 1 else config.velocity_gain
        track.velocity = tuple((1 - gain) * v + gain * (x - previous) / elapsed
                               for v, x, previous in zip(track.velocity, position, track.position))
        track.position = position
        track.last_scan = scan
        track.hits += 1
        if track.hits >= config.confirm_hits:
            outputs.append(_output(points[index], batch))

    capacity = config.max_tentative - sum(t.hits < config.confirm_hits for t in tracks.values())
    new_candidates = (i for i in range(len(points)) if i not in occupied)
    for index in nsmallest(capacity, new_candidates, key=lambda i: (-points[i]["Amp"], i)):
        tracks[next_id] = Track(positions[index], scan)
        if config.confirm_hits == 1:
            outputs.append(_output(points[index], next_id))
        next_id += 1
    state["next_id"] = next_id
    return outputs
