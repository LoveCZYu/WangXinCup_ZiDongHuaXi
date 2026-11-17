"""Causal history expansion and permutation-invariant candidate-set rejection."""
import numpy as np
import torch
from torch import nn

from experiments.competition_learning.samples import grouped_index

OLD_IN_FULL = [35, 36, 37, 38, 39, 93, 94, 97] + list(range(98, 186))


def full_pairs(pair, rows):
    """rows is the matching 101-feature history for each edge, not all rows."""
    assert pair.shape[-1] == 96 and rows.shape[-1] == 101
    result = np.concatenate((rows[..., :98], pair[..., 8:]), axis=-1)
    np.testing.assert_allclose(result[..., OLD_IN_FULL], pair, atol=0, rtol=0)
    return result.astype(np.float32)


def geometry(data):
    """Feature masks derive solely from candidate membership, never truth labels."""
    width = data['row_index'].shape[1]
    index, mask = grouped_index(data['edge_row'], len(data['row']), width)
    counts = np.bincount(data['edge_row'], minlength=len(index))
    assert np.all(counts <= width), 'Do not silently truncate geometric candidates'
    unknown = np.bincount(data['edge_row'][~data['edge_known']], minlength=len(index))
    supervised = data['row_allowed'] & (unknown == 0) & mask.any(1)
    positive = (data['edge_positive'][index] & mask).any(1)
    return index, mask, supervised, positive


class Identity(nn.Module):
    def __init__(self, full):
        super().__init__()
        self.full = full
        size = 186 if full else 96
        self.register_buffer('pair_mean', torch.zeros(size))
        self.register_buffer('pair_std', torch.ones(size))
        self.pair = nn.Sequential(nn.Linear(size, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def initialize(self, prior, training):
        mapping = OLD_IN_FULL if self.full else list(range(96))
        with torch.no_grad():
            self.pair_mean.copy_(training.mean(0))
            self.pair_std.copy_(training.std(0).clamp_min(.015))
            self.pair_mean[mapping] = prior.pair_mean
            self.pair_std[mapping] = prior.pair_std
            self.pair[0].weight.zero_()
            self.pair[0].weight[:, mapping] = prior.pair[0].weight
            self.pair[0].bias.copy_(prior.pair[0].bias)
            for i in (2, 4):self.pair[i].load_state_dict(prior.pair[i].state_dict())

    def edges(self, features):
        return self.pair(((features-self.pair_mean)/self.pair_std).clamp(-8, 8)).squeeze(-1)

    def rejection(self, rows):
        return rows.new_zeros(rows.shape[:-1])

    @torch.inference_mode()
    def score(self, features):
        return self.edges(torch.as_tensor(features, device=self.pair_mean.device)).cpu().numpy()


class Rejection(nn.Module):
    def __init__(self, context):
        super().__init__()
        self.context = context
        self.register_buffer('pair_mean', torch.zeros(96))
        self.register_buffer('pair_std', torch.ones(96))
        self.register_buffer('row_mean', torch.zeros(101))
        self.register_buffer('row_std', torch.ones(101))
        self.encoder = nn.Sequential(nn.Linear(96, 32), nn.ReLU())
        # Same decoder capacity in history and candidate conditions.
        self.null = nn.Sequential(nn.Linear(166, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.null[-1].weight)
        nn.init.zeros_(self.null[-1].bias)

    def initialize(self, prior, training_rows):
        with torch.no_grad():
            self.pair_mean.copy_(prior.pair_mean)
            self.pair_std.copy_(prior.pair_std)
            self.row_mean.copy_(training_rows.mean(0))
            self.row_std.copy_(training_rows.std(0).clamp_min(.015))

    def forward(self, rows, edges, mask):
        assert edges.shape[:-1] == mask.shape
        history = ((rows-self.row_mean)/self.row_std).clamp(-8, 8)
        context = history.new_zeros((*history.shape[:-1], 65))
        if self.context:
            z = self.encoder(((edges-self.pair_mean)/self.pair_std).clamp(-8, 8))
            count = mask.sum(-1, keepdim=True)
            mean = (z*mask[..., None]).sum(-2)/count.clamp_min(1)
            maximum = z.masked_fill(~mask[..., None], -1e4).amax(-2)
            maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
            context = torch.cat((mean, maximum, torch.log1p(count.float())/3.), -1)
        return self.null(torch.cat((history, context), -1)).squeeze(-1)
