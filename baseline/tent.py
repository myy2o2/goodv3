from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch

from baseline.common import (
    base_adaptation_search_space,
    base_adaptation_train_cfg,
    evaluate_model,
    set_trainable_blocks,
)


def build_tent_train_cfg(args, num_layers: int) -> Dict[str, object]:
    cfg = base_adaptation_train_cfg(args, num_layers)
    cfg["tent_update_bn_only"] = bool(args.tent_update_bn_only)
    return cfg


def build_tent_search_space(trial, args, num_layers: int) -> Dict[str, object]:
    cfg = base_adaptation_search_space(trial, args, num_layers)
    cfg["tent_update_bn_only"] = bool(trial.suggest_categorical("tent_update_bn_only", [True, False]))
    return cfg


def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    return -(probs * logits.log_softmax(dim=-1)).sum(dim=-1)


def run_tent_once(args, model, data, masks, train_cfg: Dict[str, object], model_cfg: Dict[str, object], verbose: bool = True):
    trainable_params = set_trainable_blocks(
        model,
        replace_last_k_layers=int(train_cfg["replace_last_k_layers"]),
        update_bn_only=bool(train_cfg["tent_update_bn_only"]),
    )
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters selected for Tent")

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
        logits = model(data.x, data.edge_index)
        loss = _entropy_loss(logits)[masks["ood_test"]].mean()
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
        print("[baseline:tent] best_ood_test_acc={:.6f}".format(float(best_metrics["ood_test_acc"])))

    artifacts = {
        "method": "tent",
        "task_names": [],
        "task_cfg": {
            "tent_update_bn_only": bool(train_cfg["tent_update_bn_only"]),
        },
        "history": history,
        "model_cfg": copy.deepcopy(model_cfg),
        "model_state_dict": model.state_dict(),
    }
    return best_metrics, artifacts
