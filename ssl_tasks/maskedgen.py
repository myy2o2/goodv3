from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import model_embeddings


class MaskedFeatureGenerationTask(nn.Module):
    name = "maskedgen"

    def __init__(self, hidden_dim: int, input_dim: int, mask_ratio: float = 0.3):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def compute_node_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        ratio = min(max(self.mask_ratio, 1e-3), 0.95)
        mask = (torch.rand_like(feat) < ratio).float()
        feat_masked = feat * (1.0 - mask)

        h = model_embeddings(model, feat_masked, edge_index, edge_weight=edge_weight)
        recon = self.decoder(h)

        sq_err = (recon - feat).pow(2) * mask
        denom = mask.sum(dim=-1).clamp_min(1.0)
        return sq_err.sum(dim=-1) / denom

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        return self.compute_node_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs).mean()
