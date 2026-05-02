from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import aggregate_edge_losses_to_nodes, model_embeddings, negative_cosine_per_example


class EdgeReconstructionTask(nn.Module):
    name = "recon"
    aliases = ("generative", "edge_reconstruction", "reconstruction")

    def compute_edge_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        embed = model.get_embed(feat, edge_index, edge_weight)
        return negative_cosine_per_example(embed[edge_index[0]], embed[edge_index[1]])

    def compute_node_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        edge_loss = self.compute_edge_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs)
        return aggregate_edge_losses_to_nodes(edge_index, edge_loss, num_nodes=feat.shape[0], reduce_by="src")

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        edge_loss = self.compute_edge_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs)
        return edge_loss.mean()


def _sample_negatives(edge_index: torch.Tensor, num_nodes: int, num_neg: int, device: torch.device) -> torch.Tensor:
    oversample = int(num_neg * 1.5) + 16
    src = torch.randint(0, num_nodes, (oversample,))
    dst = torch.randint(0, num_nodes, (oversample,))
    valid = src != dst
    src, dst = src[valid], dst[valid]
    if src.size(0) < num_neg:
        repeat = (num_neg // max(src.size(0), 1)) + 1
        src = src.repeat(repeat)
        dst = dst.repeat(repeat)
    return torch.stack([src[:num_neg], dst[:num_neg]]).to(device)


class NegativeSamplingEdgeReconstructionTask(nn.Module):
    name = "generative"

    def __init__(self, hidden_dim: int, neg_ratio: float = 1.0):
        super().__init__()
        self.neg_ratio = float(neg_ratio)
        self.transform = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.transform.weight)

    def _score(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_t = self.transform(h)
        return (h_t[edge_index[0]] * h[edge_index[1]]).sum(dim=-1)

    def compute_node_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        h = model_embeddings(model, feat, edge_index, edge_weight)
        num_nodes = h.shape[0]
        num_pos = edge_index.size(1)
        num_neg = max(1, int(num_pos * self.neg_ratio))
        neg_edge = _sample_negatives(edge_index, num_nodes, num_neg, h.device)
        pos_loss = F.binary_cross_entropy_with_logits(
            self._score(h, edge_index),
            torch.ones(num_pos, device=h.device),
            reduction="none",
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            self._score(h, neg_edge),
            torch.zeros(num_neg, device=h.device),
            reduction="none",
        )
        node_loss = torch.zeros(num_nodes, device=h.device)
        node_count = torch.zeros(num_nodes, device=h.device)
        node_loss.scatter_add_(0, edge_index[0], pos_loss)
        node_count.scatter_add_(0, edge_index[0], torch.ones(num_pos, device=h.device))
        node_loss.scatter_add_(0, neg_edge[0], neg_loss)
        node_count.scatter_add_(0, neg_edge[0], torch.ones(num_neg, device=h.device))
        return node_loss / (node_count + 1e-8)

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        return self.compute_node_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs).mean()
