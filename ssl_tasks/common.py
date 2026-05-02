from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge


def model_embeddings(model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None) -> torch.Tensor:
    if hasattr(model, "convs") and len(getattr(model, "convs", [])) <= 1:
        return model(feat, edge_index, edge_weight)
    if hasattr(model, "get_embed"):
        return model.get_embed(feat, edge_index, edge_weight)
    return model(feat, edge_index, edge_weight)


def cosine_similarity_per_example(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


def negative_cosine_per_example(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return -cosine_similarity_per_example(a, b)


def margin_penalty_per_example(a: torch.Tensor, b: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    return F.relu(cosine_similarity_per_example(a, b) - float(margin))


def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return -(a * b).sum(dim=-1).mean()


def inner_margin(a: torch.Tensor, b: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    cosine = (a * b).sum(dim=-1)
    return F.relu(cosine - float(margin)).mean()


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    return -(probs * logits.log_softmax(dim=-1)).sum(dim=-1)


def augment(model, feat: torch.Tensor, strategy: str, edge_index: torch.Tensor, p: float = 0.0, edge_weight=None):
    strategy = str(strategy).lower()
    if strategy == "dropedge":
        aug_edge_index, keep_mask = dropout_edge(edge_index, p=p, force_undirected=True, training=True)
        aug_edge_weight = edge_weight[keep_mask] if edge_weight is not None else None
        return model_embeddings(model, feat, aug_edge_index, aug_edge_weight)
    if strategy == "dropnode":
        keep = torch.bernoulli(torch.full((feat.shape[0], 1), 1.0 - p, device=feat.device))
        return model_embeddings(model, feat * keep, edge_index, edge_weight)
    if strategy == "shuffle":
        shuffled = feat[torch.randperm(feat.shape[0], device=feat.device)]
        return model_embeddings(model, shuffled, edge_index, edge_weight)
    return model_embeddings(model, feat, edge_index, edge_weight)


def aggregate_edge_losses_to_nodes(
    edge_index: torch.Tensor,
    edge_loss: torch.Tensor,
    num_nodes: int,
    reduce_by: str = "src",
) -> torch.Tensor:
    node_ids = edge_index[0] if reduce_by == "src" else edge_index[1]
    node_loss = torch.zeros(num_nodes, device=edge_loss.device)
    node_count = torch.zeros(num_nodes, device=edge_loss.device)
    node_loss.scatter_add_(0, node_ids, edge_loss)
    node_count.scatter_add_(0, node_ids, torch.ones_like(edge_loss))
    return node_loss / node_count.clamp_min(1.0)


def neighbor_mean(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    num_nodes = h.shape[0]
    src, dst = edge_index
    neigh_sum = torch.zeros_like(h)
    neigh_cnt = torch.zeros((num_nodes, 1), device=h.device)
    neigh_sum.index_add_(0, dst, h[src])
    neigh_cnt.index_add_(0, dst, torch.ones((edge_index.shape[1], 1), device=h.device))
    return neigh_sum / neigh_cnt.clamp_min(1e-8)


def weighted_neighbor_average(h: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    out = torch.zeros_like(h)
    denom = torch.zeros((h.shape[0], 1), device=h.device)
    weighted_messages = h[src] * edge_weight.unsqueeze(-1)
    out.index_add_(0, dst, weighted_messages)
    denom.index_add_(0, dst, edge_weight.unsqueeze(-1))
    return out / denom.clamp_min(1e-8)
