"""Small causal association head; paper-inspired, not a paper reproduction."""
import torch
from torch import nn


def pool(z, mask):
    count = mask.sum(-1, keepdim=True)
    mean = (z*mask[..., None]).sum(-2)/count.clamp_min(1)
    maximum = z.masked_fill(~mask[..., None], -1e4).amax(-2)
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum, torch.log1p(count.float())/3.), -1)


def pool_groups(z, groups, size):
    """All current geometric edges, including edges without training labels."""
    groups = groups.to(dtype=torch.long)
    total = z.new_zeros(size, z.shape[-1])
    count = z.new_zeros(size, 1)
    maximum = z.new_full((size, z.shape[-1]), -1e4)
    total.index_add_(0, groups, z)
    count.index_add_(0, groups, z.new_ones(len(z), 1))
    maximum.scatter_reduce_(0, groups[:, None].expand_as(z), z, reduce='amax', include_self=True)
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    return torch.cat((total/count.clamp_min(1), maximum, torch.log1p(count)/3.), -1)


class Association(nn.Module):
    def __init__(self, prior, context):
        super().__init__()
        self.prior = prior.requires_grad_(False)
        self.context = context
        self.register_buffer('row_mean', torch.zeros(101))
        self.register_buffer('row_std', torch.ones(101))
        self.encoder = nn.Sequential(nn.Linear(96, 32), nn.ReLU())
        self.residual = nn.Sequential(nn.Linear(162, 64), nn.ReLU(), nn.Linear(64, 1))
        self.null = nn.Sequential(nn.Linear(166, 32), nn.ReLU(), nn.Linear(32, 1))
        for head in (self.residual, self.null):
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)

    def initialize(self, training_rows):
        with torch.no_grad():
            self.row_mean.copy_(training_rows.mean(0))
            self.row_std.copy_(training_rows.std(0).clamp_min(.015))

    def encode(self, x):
        return self.encoder(((x-self.prior.pair_mean)/self.prior.pair_std).clamp(-8, 8))

    def edge_logits(self, pair, encoded, row_context, col_context, base=None):
        if not self.context:
            row_context = torch.zeros_like(row_context)
            col_context = torch.zeros_like(col_context)
        base = self.prior.edges(pair) if base is None else base
        return base+self.residual(torch.cat((encoded, row_context, col_context), -1)).squeeze(-1)

    def null_logits(self, rows, context):
        if not self.context:context = torch.zeros_like(context)
        history = ((rows-self.row_mean)/self.row_std).clamp(-8, 8)
        return self.null(torch.cat((history,context), -1)).squeeze(-1)

    def edges(self, pair, row_pair, row_mask, col_pair, col_mask, base=None):
        encoded = self.encode(pair)
        if self.context:
            rp, cp = pool(self.encode(row_pair),row_mask),pool(self.encode(col_pair),col_mask)
        else:
            rp, cp = encoded.new_zeros(len(pair),65),encoded.new_zeros(len(pair),65)
        return self.edge_logits(pair,encoded,rp,cp,base)

    def rejection(self, rows, row_pair, row_mask):
        context = pool(self.encode(row_pair),row_mask) if self.context else rows.new_zeros(len(rows),65)
        return self.null_logits(rows,context)

    def graph(self, pair, rows, edge_row, edge_col):
        """One hop: shared edge encoding → row/column pools → edge and null."""
        encoded = self.encode(pair)
        if self.context:
            rp = pool_groups(encoded,edge_row,len(rows))
            columns = int(edge_col.max())+1 if len(edge_col) else 0
            cp = pool_groups(encoded,edge_col,columns)
        else:
            rp = encoded.new_zeros(len(rows),65)
            cp = encoded.new_zeros(int(edge_col.max())+1 if len(edge_col) else 0,65)
        return (self.edge_logits(pair,encoded,rp[edge_row],cp[edge_col]),self.null_logits(rows,rp))
