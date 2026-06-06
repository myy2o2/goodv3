from __future__ import annotations

import copy
from typing import Dict

from baseline.common import base_adaptation_train_cfg, evaluate_model


def build_stage1_train_cfg(args, num_layers: int) -> Dict[str, object]:
    return base_adaptation_train_cfg(args, num_layers)


def build_stage1_search_space(trial, args, num_layers: int) -> Dict[str, object]:
    return build_stage1_train_cfg(args, num_layers)


def run_stage1_once(args, model, data, masks, train_cfg: Dict[str, object], model_cfg: Dict[str, object], verbose: bool = True):
    metrics = evaluate_model(model, data, masks)
    history = [{"epoch": 0, **metrics}]
    if verbose:
        print("[baseline:stage1] ood_test_acc={:.6f}".format(float(metrics["ood_test_acc"])))

    artifacts = {
        "method": "stage1",
        "task_names": [],
        "task_cfg": {},
        "history": history,
        "model_cfg": copy.deepcopy(model_cfg),
        "model_state_dict": model.state_dict(),
    }
    return metrics, artifacts
