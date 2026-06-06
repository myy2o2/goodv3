from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

from baseline.common import base_adaptation_search_space, base_adaptation_train_cfg, evaluate_model


def _cosine_per_example(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


def _inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (1.0 - _cosine_per_example(a, b)).mean()


def _inner_margin(a: torch.Tensor, b: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    cosine = _cosine_per_example(a, b)
    return F.relu(cosine - float(margin)).mean()


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    return -(probs * logits.log_softmax(dim=-1)).sum(dim=-1)


def _project(values: torch.Tensor, budget: int, eps: float = 1e-7) -> torch.Tensor:
    values = values.clone()
    if torch.clamp(values, 0, 1).sum() > budget:
        left = float((values - 1).min().item())
        right = float(values.max().item())

        def func(x: float) -> torch.Tensor:
            return torch.clamp(values - x, 0, 1).sum() - budget

        for _ in range(128):
            mid = 0.5 * (left + right)
            if float(func(mid).item()) == 0.0:
                break
            if float((func(mid) * func(left)).item()) < 0:
                right = mid
            else:
                left = mid
            if (right - left) <= 1e-5:
                break
        miu = 0.5 * (left + right)
        values = torch.clamp(values - miu, min=eps, max=1 - eps)
    else:
        values = torch.clamp(values, min=eps, max=1 - eps)
    return values


def _augment(model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, strategy: str = "dropedge", p: float = 0.05):
    strategy = str(strategy).lower()
    if strategy == "shuffle":
        feat = feat[torch.randperm(feat.shape[0], device=feat.device)]
        return model.get_embed(feat, edge_index, edge_weight=edge_weight)
    if strategy == "dropnode":
        mask = (torch.rand(feat.shape[0], 1, device=feat.device) > p).float()
        feat = feat * mask
        return model.get_embed(feat, edge_index, edge_weight=edge_weight)
    if strategy == "rwsample":
        strategy = "dropedge"
    if strategy == "dropedge":
        edge_index_aug, edge_mask = dropout_edge(edge_index, p=p, force_undirected=False, training=True)
        edge_weight_aug = edge_weight[edge_mask] if edge_weight is not None else None
        return model.get_embed(feat, edge_index_aug, edge_weight=edge_weight_aug)
    return model.get_embed(feat, edge_index, edge_weight=edge_weight)


def build_gtrans_train_cfg(args, num_layers: int) -> Dict[str, object]:
    cfg = base_adaptation_train_cfg(args, num_layers)
    cfg.update(
        {
            "gtrans_loss": str(args.gtrans_loss),
            "gtrans_strategy": str(args.gtrans_strategy),
            "gtrans_margin": float(args.gtrans_margin),
            "gtrans_ratio": float(args.gtrans_ratio),
            "gtrans_loop_feat": int(args.gtrans_loop_feat),
            "gtrans_loop_adj": int(args.gtrans_loop_adj),
        }
    )
    return cfg


def build_gtrans_search_space(trial, args, num_layers: int) -> Dict[str, object]:
    cfg = base_adaptation_search_space(trial, args, num_layers)
    cfg.update(
        {
            "ssl_lr": trial.suggest_float("ssl_lr", 1e-5, 1e-2, log=True),
            "gtrans_ratio": trial.suggest_float("gtrans_ratio", 0.01, 0.3, log=True),
            "gtrans_margin": trial.suggest_float("gtrans_margin", -1.0, 0.5),
            "gtrans_loop_feat": int(trial.suggest_int("gtrans_loop_feat", 1, 6)),
            "gtrans_loop_adj": int(trial.suggest_int("gtrans_loop_adj", 0, 3)),
            "gtrans_loss": str(args.gtrans_loss),
            "gtrans_strategy": str(args.gtrans_strategy),
        }
    )
    return cfg


def _test_time_loss(train_cfg: Dict[str, object], model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None):
    loss = torch.tensor(0.0, device=feat.device)
    loss_spec = str(train_cfg["gtrans_loss"]).lower()

    if "lc" in loss_spec:
        strategy = str(train_cfg["gtrans_strategy"]).lower()
        margin = float(train_cfg["gtrans_margin"])
        output1 = _augment(model, feat, edge_index, edge_weight=edge_weight, strategy=strategy, p=0.05)
        output2 = _augment(model, feat, edge_index, edge_weight=edge_weight, strategy="dropedge", p=0.0)
        output3 = _augment(model, feat, edge_index, edge_weight=edge_weight, strategy="shuffle", p=0.0)
        if margin != -1.0:
            loss = loss + _inner(output1, output2) - _inner_margin(output2, output3, margin=margin)
        else:
            loss = loss + _inner(output1, output2) - _inner(output2, output3)

    if "recon" in loss_spec:
        output2 = model.get_embed(feat, edge_index, edge_weight=edge_weight)
        loss = loss + _inner(output2[edge_index[0]], output2[edge_index[1]])

    if "entropy" in loss_spec:
        logits = model.forward(feat, edge_index, edge_weight=edge_weight)
        batch_size = min(1000, logits.shape[0])
        sampled = torch.randperm(logits.shape[0], device=logits.device)[:batch_size]
        loss = loss + _softmax_entropy(logits[sampled]).mean()

    return loss


def run_gtrans_once(args, model, data, masks, train_cfg: Dict[str, object], model_cfg: Dict[str, object], verbose: bool = True):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    feat = data.x
    edge_index = data.edge_index
    budget = max(int(float(train_cfg["gtrans_ratio"]) * edge_index.shape[1] / 2), 1)
    feat_lr = float(train_cfg["encoder_lr"])
    adj_lr = float(train_cfg["ssl_lr"])
    loop_feat = max(int(train_cfg["gtrans_loop_feat"]), 0)
    loop_adj = max(int(train_cfg["gtrans_loop_adj"]), 0)

    delta_feat = torch.nn.Parameter(torch.full_like(feat, 1e-7))
    edge_remove = torch.nn.Parameter(torch.full((edge_index.shape[1],), 1e-7, device=feat.device))

    opt_feat = torch.optim.Adam([delta_feat], lr=feat_lr)
    opt_adj = torch.optim.Adam([edge_remove], lr=adj_lr)

    best_metrics = evaluate_model(model, data, masks)
    history: List[Dict[str, float]] = [{"epoch": 0, "loss": float("nan"), **best_metrics}]

    for epoch in range(1, int(train_cfg["finetune_epochs"]) + 1):
        transformed_feat = feat + delta_feat
        edge_weight = (1.0 - edge_remove).clamp_min(1e-7).clamp_max(1.0 - 1e-7)

        for _ in range(loop_feat):
            opt_feat.zero_grad()
            loss = _test_time_loss(train_cfg, model, transformed_feat, edge_index, edge_weight=edge_weight)
            if loss.requires_grad:
                loss.backward()
                opt_feat.step()
            transformed_feat = feat + delta_feat
            edge_weight = (1.0 - edge_remove).clamp_min(1e-7).clamp_max(1.0 - 1e-7)

        for _ in range(loop_adj):
            opt_adj.zero_grad()
            transformed_feat = feat + delta_feat.detach()
            edge_weight = (1.0 - edge_remove).clamp_min(1e-7).clamp_max(1.0 - 1e-7)
            loss = _test_time_loss(train_cfg, model, transformed_feat, edge_index, edge_weight=edge_weight)
            if loss.requires_grad:
                loss.backward()
                opt_adj.step()
                with torch.no_grad():
                    edge_remove.data = _project(edge_remove.data, budget=budget, eps=1e-7)

        with torch.no_grad():
            transformed_feat = feat + delta_feat
            edge_weight = (1.0 - edge_remove).clamp_min(1e-7).clamp_max(1.0 - 1e-7)
            loss = _test_time_loss(train_cfg, model, transformed_feat, edge_index, edge_weight=edge_weight)
            metrics = evaluate_model(model, data, masks, x=transformed_feat, edge_weight=edge_weight)
            history.append({"epoch": int(epoch), "loss": float(loss.item()), **metrics})
            if verbose and epoch == 1:
                print("[gtrans] epoch 1 loss={:.6f}".format(float(loss.item())))

    with torch.no_grad():
        transformed_feat = feat + delta_feat
        edge_weight = (1.0 - edge_remove).clamp_min(1e-7).clamp_max(1.0 - 1e-7)
        final_metrics = evaluate_model(model, data, masks, x=transformed_feat, edge_weight=edge_weight)

    if verbose:
        print("[gtrans] final ood_test_acc={:.6f}".format(float(final_metrics["ood_test_acc"])))

    artifacts = {
        "method": "gtrans",
        "task_names": [],
        "task_cfg": {
            "gtrans_loss": str(train_cfg["gtrans_loss"]),
            "gtrans_strategy": str(train_cfg["gtrans_strategy"]),
            "gtrans_margin": float(train_cfg["gtrans_margin"]),
            "gtrans_ratio": float(train_cfg["gtrans_ratio"]),
            "gtrans_loop_feat": int(train_cfg["gtrans_loop_feat"]),
            "gtrans_loop_adj": int(train_cfg["gtrans_loop_adj"]),
        },
        "history": history,
        "model_cfg": copy.deepcopy(model_cfg),
        "model_state_dict": model.state_dict(),
        "delta_feat": delta_feat.detach().cpu(),
        "edge_remove": edge_remove.detach().cpu(),
    }
    return final_metrics, artifacts
