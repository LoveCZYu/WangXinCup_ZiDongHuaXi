"""Shared absolute assignment score and an explicitly trained row-null head."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.competition_learning.features import PAIR_DIM, ROW_DIM


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('pair_mean', torch.zeros(PAIR_DIM))
        self.register_buffer('pair_std', torch.ones(PAIR_DIM))
        self.register_buffer('row_mean', torch.zeros(ROW_DIM))
        self.register_buffer('row_std', torch.ones(ROW_DIM))
        self.pair = nn.Sequential(nn.Linear(PAIR_DIM, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.null = nn.Sequential(nn.Linear(ROW_DIM, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.null[-1].weight)
        nn.init.zeros_(self.null[-1].bias)

    def edges(self, features):
        return self.pair(((features-self.pair_mean)/self.pair_std).clamp(-8, 8)).squeeze(-1)

    def rejection(self, features):
        return self.null(((features-self.row_mean)/self.row_std).clamp(-8, 8)).squeeze(-1)

    def initialize(self, old, training_pair, training_rows):
        """Preserve the pretrained 93-feature function; new inputs start at zero."""
        with torch.no_grad():
            self.pair_mean[:93].copy_(old.mean)
            self.pair_std[:93].copy_(old.std)
            extra = training_pair[:, 93:]
            self.pair_mean[93:].copy_(extra.mean(0))
            self.pair_std[93:].copy_(extra.std(0).clamp_min(.015))
            self.row_mean.copy_(training_rows.mean(0))
            self.row_std.copy_(training_rows.std(0).clamp_min(.015))
            self.pair[0].weight.zero_()
            self.pair[0].weight[:, :93].copy_(old.net[0].weight)
            self.pair[0].bias.copy_(old.net[0].bias)
            for index in (2, 4):self.pair[index].load_state_dict(old.net[index].state_dict())

    @torch.inference_mode()
    def score(self, pair):
        return self.edges(torch.as_tensor(pair, device=self.pair_mean.device)).cpu().numpy()


def set_loss(logits, valid, positive):
    assert valid.shape == positive.shape == logits.shape
    assert torch.all(positive.any(1)) and torch.all(~positive | valid)
    return (logits.masked_fill(~valid, -1e4).logsumexp(1)-
            logits.masked_fill(~positive, -1e4).logsumexp(1)).mean()


def objective(model, batch, joint):
    flat_x, flat_y, row_x, row_edges, row_valid, row_positive, col_edges, col_valid, col_positive, col_semantic = batch
    flat = model.edges(flat_x)
    row = model.edges(row_edges)
    col = model.edges(col_edges)
    all_logits = torch.cat((flat, row[row_valid], col[col_valid]))
    all_targets = torch.cat((flat_y, row_positive[row_valid].float(), col_semantic[col_valid].float()))
    # Column preference may select one canonical ID, while BCE still supervises
    # semantic same-identity; canonical preference is used only by the joint loss.
    binary = F.binary_cross_entropy_with_logits(all_logits, all_targets)
    if not joint:return binary
    null = model.rejection(row_x)
    rlogits = torch.cat((row, null[:, None]), 1)
    rvalid = torch.cat((row_valid, torch.ones_like(null[:, None], dtype=torch.bool)), 1)
    rtarget = torch.cat((row_positive, ~row_positive.any(1, keepdim=True)), 1)
    clogits = torch.cat((col, col.new_zeros(len(col), 1)), 1)
    cvalid = torch.cat((col_valid, torch.ones(len(col), 1, dtype=torch.bool, device=col.device)), 1)
    ctarget = torch.cat((col_positive, ~col_positive.any(1, keepdim=True)), 1)
    return binary + .5*set_loss(rlogits, rvalid, rtarget) + .5*set_loss(clogits, cvalid, ctarget)
