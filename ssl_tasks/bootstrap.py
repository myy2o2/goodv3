from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import augment


class BootstrapConsistencyTask(nn.Module):
    name = "bootstrap"

    def __init__(
        self,
        hidden_dim: int,
        edge_drop: float = 0.2,
        node_drop: float = 0.1,
        predictor_hidden: int = 0,
    ):
        super().__init__()
        self.edge_drop = float(edge_drop)
        self.node_drop = float(node_drop)
        mid = int(predictor_hidden) if int(predictor_hidden) > 0 else int(hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, hidden_dim),
        )

    def _view(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None) -> torch.Tensor:
        h_edge = augment(
            model,
            feat,
            strategy="dropedge",
            p=self.edge_drop,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        h_node = augment(
            model,
            feat,
            strategy="dropnode",
            p=self.node_drop,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        return 0.5 * (h_edge + h_node)

    def compute_node_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        h1 = self._view(model, feat, edge_index, edge_weight=edge_weight)
        h2 = self._view(model, feat, edge_index, edge_weight=edge_weight)

        z1 = self.projector(h1)
        z2 = self.projector(h2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        z1_t = F.normalize(z1.detach(), dim=-1)
        z2_t = F.normalize(z2.detach(), dim=-1)

        loss12 = 2.0 - 2.0 * (p1 * z2_t).sum(dim=-1)
        loss21 = 2.0 - 2.0 * (p2 * z1_t).sum(dim=-1)
        return 0.5 * (loss12 + loss21)

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        return self.compute_node_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs).mean()
