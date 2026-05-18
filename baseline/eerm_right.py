from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_dense_adj


def _accuracy(logits: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    if mask is None or int(mask.sum()) == 0:
        return float("nan")
    pred = logits[mask].argmax(dim=-1)
    return float((pred == y[mask]).float().mean().item())


@torch.no_grad()
def _evaluate(model, data, masks) -> Dict[str, float]:
    model.eval()
    logits = model(data.x, data.edge_index)
    return {
        "id_val_acc": _accuracy(logits, data.y, masks["id_val"]),
        "id_test_acc": _accuracy(logits, data.y, masks["id_test"]),
        "ood_val_acc": _accuracy(logits, data.y, masks["ood_val"]),
        "ood_test_acc": _accuracy(logits, data.y, masks["ood_test"]),
    }


class DenseGraphEditor(nn.Module):
    def __init__(self, num_views: int, num_nodes: int, device: torch.device):
        super().__init__()
        self.num_views = int(num_views)
        self.num_nodes = int(num_nodes)
        self.device = device
        self.B = nn.Parameter(torch.empty(self.num_views, self.num_nodes, self.num_nodes, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, adj_dense: torch.Tensor, num_sample: int, view_id: int):
        logits = self.B[int(view_id)]
        col_logprob = torch.log_softmax(logits, dim=0)
        probs_by_col = col_logprob.transpose(0, 1).exp().contiguous()
        sampled_rows = torch.multinomial(probs_by_col, num_samples=int(num_sample), replacement=True)

        toggle = torch.zeros_like(adj_dense, dtype=torch.float32)
        cols = torch.arange(self.num_nodes, device=self.device).unsqueeze(1).expand(self.num_nodes, int(num_sample))
        toggle[sampled_rows, cols] = 1.0

        edited_adj = adj_dense + toggle * ((1.0 - adj_dense) - adj_dense)
        edge_index = dense_to_sparse(edited_adj)[0]

        chosen_logprob = col_logprob[sampled_rows, cols]
        log_p = chosen_logprob.sum()
        return edge_index, log_p


def _supervised_loss(model, data, masks, edge_index: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    logits = model(data.x, edge_index)
    train_mask = masks["train"]
    if train_mask is None or int(train_mask.sum()) == 0:
        raise ValueError("eerm_right requires a non-empty train_mask")
    target = data.y[train_mask].view(-1).long()
    return criterion(logits[train_mask], target)


def _forward_eerm_right(model, editor: DenseGraphEditor, data, masks, train_cfg: Dict[str, float], criterion: nn.Module):
    adj_dense = to_dense_adj(data.edge_index, max_num_nodes=data.x.shape[0])[0].to(data.x.device).float()
    losses: List[torch.Tensor] = []
    log_p = torch.tensor(0.0, device=data.x.device)
    num_views = int(train_cfg["eerm_right_k"])
    num_sample = int(train_cfg["eerm_right_num_sample"])
    for view_id in range(num_views):
        edge_index_k, log_p_k = editor(adj_dense, num_sample=num_sample, view_id=view_id)
        loss_k = _supervised_loss(model, data, masks, edge_index_k, criterion)
        losses.append(loss_k.view(-1))
        log_p = log_p + log_p_k
    loss_vec = torch.cat(losses, dim=0)
    var, mean = torch.var_mean(loss_vec, unbiased=False)
    return var, mean, log_p


def run_eerm_right_once(args, model, data, masks, train_cfg: Dict[str, float], model_cfg: Dict[str, object], verbose: bool = True):
    num_nodes = int(data.x.shape[0])
    device = data.x.device
    num_views = int(train_cfg["eerm_right_k"])
    editor = DenseGraphEditor(num_views=num_views, num_nodes=num_nodes, device=device)
    criterion = nn.CrossEntropyLoss()

    optimizer_gnn = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["encoder_lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    optimizer_aug = torch.optim.AdamW(
        editor.parameters(),
        lr=float(train_cfg["ssl_lr"]),
        weight_decay=0.0,
    )

    best_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    best_editor_state = copy.deepcopy(editor.state_dict())
    best_metrics = _evaluate(model, data, masks)
    history: List[Dict[str, float]] = [{"epoch": 0, "var": float("nan"), "mean": float("nan"), **best_metrics}]

    num_inner = max(int(train_cfg["eerm_right_t"]), 1)
    beta = float(train_cfg["eerm_right_beta"])

    for epoch in range(1, int(train_cfg["finetune_epochs"]) + 1):
        model.train()
        editor.train()
        editor.reset_parameters()

        last_var = torch.tensor(float("nan"), device=device)
        last_mean = torch.tensor(float("nan"), device=device)

        for inner_step in range(num_inner):
            var, mean, log_p = _forward_eerm_right(model, editor, data, masks, train_cfg, criterion)
            outer_loss = var + beta * mean
            reward = var.detach()
            inner_loss = -reward * log_p

            if inner_step == 0:
                optimizer_gnn.zero_grad()
                outer_loss.backward(retain_graph=True)
                optimizer_gnn.step()

            optimizer_aug.zero_grad()
            inner_loss.backward()
            optimizer_aug.step()

            last_var = var.detach()
            last_mean = mean.detach()

        metrics = _evaluate(model, data, masks)
        history.append(
            {
                "epoch": int(epoch),
                "var": float(last_var.item()),
                "mean": float(last_mean.item()),
                **metrics,
            }
        )

        score = metrics["ood_val_acc"]
        if np.isnan(score):
            score = metrics["id_val_acc"]
        if not np.isnan(score) and score > best_score:
            best_score = float(score)
            best_state = copy.deepcopy(model.state_dict())
            best_editor_state = copy.deepcopy(editor.state_dict())
            best_metrics = dict(metrics)

    model.load_state_dict(best_state)
    editor.load_state_dict(best_editor_state)
    if verbose:
        print("[eerm_right] best_ood_val_acc={:.6f} best_ood_test_acc={:.6f}".format(float(best_metrics["ood_val_acc"]), float(best_metrics["ood_test_acc"])))

    artifacts = {
        "method": "eerm_right",
        "task_names": [],
        "task_cfg": {
            "eerm_right_k": int(train_cfg["eerm_right_k"]),
            "eerm_right_t": int(train_cfg["eerm_right_t"]),
            "eerm_right_num_sample": int(train_cfg["eerm_right_num_sample"]),
            "eerm_right_beta": float(train_cfg["eerm_right_beta"]),
        },
        "history": history,
        "model_cfg": copy.deepcopy(model_cfg),
        "model_state_dict": model.state_dict(),
        "editor_state_dict": editor.state_dict(),
    }
    return best_metrics, artifacts
