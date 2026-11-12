"""Causal B02 windows, transfer initialization and original-point association."""
from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F

from track_mt3.data.window import MeasurementWindow
from track_mt3.models import TrackMT3
from track_mt3.tracking import OnlineTracker
from track_mt3.losses.criterion import CollectiveAverageCriterion

FEATURES = ('Azimuth', 'Distance', 'Elevation')
GATE = (.025, .020, .065)  # Same physical feature ordering as team's baseline.
HISTORY = 6
MAX_POINTS = 512
QUERIES = 96


def coordinates(points):
    return np.asarray([[p[k] for k in FEATURES] for p in points], dtype=np.float32).reshape(-1, 3)


def select_points(points, maximum=MAX_POINTS):
    """Bound tokens without labels: deterministic evenly spaced original point IDs."""
    ordered = sorted(points, key=lambda p: p['PntID'])
    if len(ordered) <= maximum:
        return ordered
    indices = np.linspace(0, len(ordered)-1, maximum, dtype=np.int64)
    return [ordered[i] for i in indices]


@dataclass(frozen=True)
class B02Window(MeasurementWindow):
    current_positions: torch.Tensor = torch.empty((0, 3))
    current_labels: torch.Tensor = torch.empty(0, dtype=torch.long)


def make_window(history, circle, labels=None, ages=None):
    """history holds (circle, original points, optional supervision)."""
    parts, times, ids = [], [], []
    for index, points, truth in history:
        selected = select_points(points)
        parts.append(coordinates(selected))
        times.extend([19-(circle-index)]*len(selected))
        # Inference never receives labels. Known clutter and unknown are distinct.
        ids.extend([(truth[p['PntID']] if truth[p['PntID']] > 0 else
                     -1 if truth[p['PntID']] == 0 else -2) if truth is not None else -2
                    for p in selected])
    current = history[-1][1]
    targets = [p for p in current if labels is not None and labels[p['PntID']] > 0]
    return B02Window(circle,
        torch.from_numpy(np.concatenate(parts) if parts else np.empty((0, 3), np.float32)),
        torch.tensor(times, dtype=torch.long), torch.tensor(ids, dtype=torch.long),
        torch.from_numpy(coordinates(targets)),
        torch.tensor([labels[p['PntID']] for p in targets], dtype=torch.long),
        torch.tensor([(ages or {}).get(labels[p['PntID']], 0) for p in targets], dtype=torch.long),
        torch.from_numpy(coordinates(current)),
        torch.tensor([labels[p['PntID']] for p in current] if labels is not None else [], dtype=torch.long))


class IgnoreUnknownCriterion(CollectiveAverageCriterion):
    def set_window(self, window):
        self.current = window

    def _layer_loss(self, positions, logits, targets, assignment, target_ages=None):
        localization, _ = super()._layer_loss(positions, logits, targets, assignment, target_ages)
        # Keep all original inputs. Only mask a negative query's confidence loss
        # if its nearest current measurement is unknown and within the output gate.
        labels = torch.zeros_like(logits)
        labels[assignment.prediction_indices] = 1
        keep = torch.ones(len(positions), dtype=torch.bool, device=positions.device)
        if len(self.current.current_positions):
            measured = self.current.current_positions.to(positions.device)
            truth = self.current.current_labels.to(positions.device)
            distance = torch.cdist(positions.detach()/positions.new_tensor(GATE), measured/positions.new_tensor(GATE), p=float('inf'))
            minimum, nearest = distance.min(dim=1)
            keep &= ~((truth[nearest] == -1) & (minimum <= 1))
        keep[assignment.prediction_indices] = True
        weights = torch.full_like(logits, self.config.confidence_positive_weight)
        weights[assignment.newborn_prediction_indices] *= self.config.newborn_weight
        confidence = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weights, reduction='none').flatten()
        confidence = confidence[keep].mean() if keep.any() else logits.sum()*0
        return localization, confidence


def make_model(upstream, device='cuda'):
    from track_mt3.config_merge import load_merged_config
    c = load_merged_config(upstream/'configs/paper.yaml', upstream/'configs/training/track_mt3.yaml')
    c.model.measurement_dim = c.model.output_dim = 3
    c.simulation.field_of_view = (0., 1., 0., 1., 0., 1.)
    c.model.num_detection_queries = c.model.max_tracks = QUERIES
    c.model.detection_reference_init = 'grid'
    c.loss.localization_weight = c.loss.matching_state_weight = 20.
    return TrackMT3(c).to(device)


def transfer(model, path):
    old = torch.load(path, map_location='cpu')['model']
    new = model.state_dict()
    copied, partial, fresh = [], [], []
    for key, dst in new.items():
        src = old.get(key)
        if key.startswith('normalizer.') or key == 'detection_reference_offset':
            fresh.append(key)
        elif src is not None and src.shape == dst.shape:
            dst.copy_(src); copied.append(key)
        elif key == 'qtm.prediction_to_query.layers.0.weight':
            dst.zero_(); dst[:, :2].copy_(src[:, :2]); dst[:, 3].copy_(src[:, 2]); partial.append(key)
        elif key == 'detection_queries':
            dst.copy_(src.to(dst.device).repeat((len(dst)+len(src)-1)//len(src), 1)[:len(dst)] + .002*torch.randn_like(dst)); partial.append(key)
        elif src is not None and src.ndim == dst.ndim and all(a <= b for a,b in zip(src.shape, dst.shape)):
            # Added elevation inputs/outputs start at zero; pretrained first two coordinates survive.
            dst.zero_(); dst[tuple(slice(0,n) for n in src.shape)].copy_(src); partial.append(key)
        else:
            fresh.append(key)
    model.load_state_dict(new, strict=True)
    return {'copied_tensors':copied, 'expanded_tensors':partial, 'fresh_tensors':fresh,
            'copied_parameter_elements':sum(old[k].numel() for k in copied if k in dict(model.named_parameters())),
            'total_parameters':sum(p.numel() for p in model.parameters())}


def associate(positions, track_ids, points):
    """Gated one-to-one matching with dummy columns; never return synthetic points."""
    if not len(positions) or not points:
        return []
    measured = coordinates(points)
    delta = np.abs((np.asarray(positions)[:, None, :]-measured[None, :, :])/np.asarray(GATE))
    valid = (delta <= 1).all(axis=2)
    cost = np.linalg.norm(delta, axis=2)
    n = len(positions)
    matrix = np.concatenate([np.where(valid, cost, 1e6), np.full((n,n), 10.)], axis=1)
    rows, cols = linear_sum_assignment(matrix)
    return [{**points[j], 'BatchID':int(track_ids[i])+1}
            for i,j in zip(rows,cols) if j < len(points) and valid[i,j]]


class Handler:
    def __init__(self, model):
        self.model = model
        self.telemetry = {'circles':0, 'raw_tracks':0, 'matched_original_points':0,
                          'input_points':0, 'retained_input_points':0}

    @torch.no_grad()
    def __call__(self, points, state):
        if 'tracker' not in state:
            state.update(tracker=OnlineTracker(self.model), history=deque(maxlen=HISTORY), circle=-1)
        state['circle'] += 1
        state['history'].append((state['circle'], points, None))
        window = make_window(state['history'], state['circle'])
        output = state['tracker'].step(window)
        result = associate(output.positions.numpy(), output.track_ids.numpy(), points)
        self.telemetry['circles'] += 1
        self.telemetry['raw_tracks'] += len(output.positions)
        self.telemetry['matched_original_points'] += len(result)
        self.telemetry['input_points'] += len(points)
        self.telemetry['retained_input_points'] += len(select_points(points))
        return result
