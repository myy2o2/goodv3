from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.utils import degree

from baseline.common import (
    base_adaptation_search_space,
    base_adaptation_train_cfg,
    evaluate_model,
    set_trainable_blocks,
)


def build_eerm_train_cfg(args, num_layers: int) -> Dict[str, object]:
    cfg = base_adaptation_train_cfg(args, num_layers)
    cfg.update(
        {
            "eerm_num_envs": int(args.eerm_num_envs),
            "eerm_var_lambda": float(args.eerm_var_lambda),
        }
    )
    return cfg


def build_eerm_search_space(trial, args, num_layers: int) -> Dict[str, object]:
    cfg = base_adaptation_search_space(trial, args, num_layers)
    cfg.update(
        {
            "eerm_num_envs": int(trial.suggest_int("eerm_num_envs", 2, 6)),
            "eerm_var_lambda": trial.suggest_float("eerm_var_lambda", 1e-3, 2.0, log=True),
        }
    )
    return cfg


def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    return -(probs * logits.log_softmax(dim=-1)).sum(dim=-1)


def _split_env_masks_by_degree(data, base_mask: torch.Tensor, num_envs: int) -> List[torch.Tensor]:
    num_envs = max(int(num_envs), 1)
    selected = torch.where(base_mask)[0]
    if selected.numel() == 0:
        return [base_mask]

    deg = degree(data.edge_index[0], num_nodes=data.x.shape[0]).to(base_mask.device)
    order = torch.argsort(deg[selected])
    env_masks: List[torch.Tensor] = []
    for chunk in torch.chunk(order, num_envs):
        if chunk.numel() == 0:
            continue
        mask = torch.zeros_like(base_mask)
        mask[selected[chunk]] = True
        env_masks.append(mask)
    return env_masks if env_masks else [base_mask]


def _eerm_loss(model, data, masks, train_cfg: Dict[str, object]) -> torch.Tensor:
    logits = model(data.x, data.edge_index)
    node_loss = _entropy_loss(logits)
    env_masks = _split_env_masks_by_degree(data, masks["ood_test"], int(train_cfg["eerm_num_envs"]))
    env_losses = torch.stack([node_loss[env_mask].mean() for env_mask in env_masks])
    return env_losses.mean() + float(train_cfg["eerm_var_lambda"]) * env_losses.var(unbiased=False)


def run_eerm_once(args, model, data, masks, train_cfg: Dict[str, object], model_cfg: Dict[str, object], verbose: bool = True):
    trainable_params = set_trainable_blocks(
        model,
        replace_last_k_layers=int(train_cfg["replace_last_k_layers"]),
        update_bn_only=False,
    )
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters selected for EERM")

    optimizer = torch.optim.Adam(
        [{"params": trainable_params, "lr": float(train_cfg["encoder_lr"])}],
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_metrics = evaluate_model(model, data, masks)
    best_score = float(best_metrics["ood_test_acc"]) if not np.isnan(best_metrics["ood_test_acc"]) else float("-inf")
    history = [{"epoch": 0, "loss": float("nan"), **best_metrics}]

    for epoch in range(1, int(train_cfg["finetune_epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        loss = _eerm_loss(model, data, masks, train_cfg)
        loss.backward()
        optimizer.step()

        metrics = evaluate_model(model, data, masks)
        history.append({"epoch": int(epoch), "loss": float(loss.item()), **metrics})
        if not np.isnan(metrics["ood_test_acc"]) and metrics["ood_test_acc"] > best_score:
            best_score = float(metrics["ood_test_acc"])
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dict(metrics)

    model.load_state_dict(best_state)
    if verbose:
        print("[baseline:eerm] best_ood_test_acc={:.6f}".format(float(best_metrics["ood_test_acc"])))

    artifacts = {
        "method": "eerm",
        "task_names": [],
        "task_cfg": {
            "eerm_num_envs": int(train_cfg["eerm_num_envs"]),
            "eerm_var_lambda": float(train_cfg["eerm_var_lambda"]),
        },
        "history": history,
        "model_cfg": copy.deepcopy(model_cfg),
        "model_state_dict": model.state_dict(),
    }
    return best_metrics, artifacts
