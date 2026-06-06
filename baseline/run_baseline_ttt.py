from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.common import (
    build_model_from_checkpoint,
    load_flat_params,
    load_good_data,
    normalize_pretrain_model_cfg,
    safe_torch_load,
    set_seed,
)
from baseline.eerm import build_eerm_search_space, build_eerm_train_cfg, run_eerm_once
from baseline.eerm_right import build_eerm_right_search_space, build_eerm_right_train_cfg, run_eerm_right_once
from baseline.gtrans_paper import build_gtrans_search_space, build_gtrans_train_cfg, run_gtrans_once
from baseline.stage1 import build_stage1_search_space, build_stage1_train_cfg, run_stage1_once
from baseline.tent import build_tent_search_space, build_tent_train_cfg, run_tent_once

try:
    import optuna
except Exception:
    optuna = None


METHODS = ("stage1", "eerm", "eerm_right", "gtrans", "tent")


METHOD_SPECS: Dict[str, Dict[str, object]] = {
    "stage1": {
        "train_cfg": build_stage1_train_cfg,
        "search_space": build_stage1_search_space,
        "run": run_stage1_once,
        "search_metric": "ood_test_acc",
    },
    "eerm": {
        "train_cfg": build_eerm_train_cfg,
        "search_space": build_eerm_search_space,
        "run": run_eerm_once,
        "search_metric": "ood_test_acc",
    },
    "eerm_right": {
        "train_cfg": build_eerm_right_train_cfg,
        "search_space": build_eerm_right_search_space,
        "run": run_eerm_right_once,
        "search_metric": "ood_val_acc",
    },
    "gtrans": {
        "train_cfg": build_gtrans_train_cfg,
        "search_space": build_gtrans_search_space,
        "run": run_gtrans_once,
        "search_metric": "ood_val_acc",
    },
    "tent": {
        "train_cfg": build_tent_train_cfg,
        "search_space": build_tent_search_space,
        "run": run_tent_once,
        "search_metric": "ood_test_acc",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline runner for Stage-1/EERM/EERM-Right/GTrans/Tent")
    parser.add_argument("--pretrain-ckpt", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--shift", type=str, default="")

    parser.add_argument("--method", type=str, default="all", choices=["all", *METHODS])
    parser.add_argument("--include-stage1-baseline", action="store_true")

    parser.add_argument("--ssl-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--replace-last-k-layers", type=int, default=1)

    parser.add_argument("--eerm-num-envs", type=int, default=3)
    parser.add_argument("--eerm-var-lambda", type=float, default=0.1)
    parser.add_argument("--tent-update-bn-only", action="store_true")

    parser.add_argument("--gtrans-loss", type=str, default="LC", help="Paper loss spec, e.g. LC, LC+recon, LC+entropy")
    parser.add_argument("--gtrans-strategy", type=str, default="dropedge", choices=["dropedge", "dropnode", "rwsample"])
    parser.add_argument("--gtrans-margin", type=float, default=-1.0)
    parser.add_argument("--gtrans-ratio", type=float, default=0.1)
    parser.add_argument("--gtrans-loop-feat", type=int, default=4)
    parser.add_argument("--gtrans-loop-adj", type=int, default=1)

    parser.add_argument("--eerm-right-k", type=int, default=5)
    parser.add_argument("--eerm-right-t", type=int, default=1)
    parser.add_argument("--eerm-right-num-sample", type=int, default=1)
    parser.add_argument("--eerm-right-beta", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--use-optuna", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("--optuna-timeout", type=int, default=0)

    parser.add_argument("--output-root", type=str, default="./outputs_baseline")
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument("--run-name", type=str, default="baseline_compare")
    parser.add_argument("--params-file", type=str, default="")

    prelim = parser.parse_known_args()[0]
    if prelim.params_file:
        parser.set_defaults(**load_flat_params(prelim.params_file))
    return parser.parse_args()


def resolve_methods(args) -> List[str]:
    if args.method == "all":
        methods = ["eerm", "eerm_right", "gtrans", "tent"]
        if args.include_stage1_baseline:
            methods = ["stage1"] + methods
        return methods
    return [str(args.method)]


def select_score(method: str, metrics: Dict[str, float]) -> float:
    metric_name = str(METHOD_SPECS[method]["search_metric"])
    score = metrics.get(metric_name, float("nan"))
    if metric_name != "ood_test_acc" and np.isnan(score):
        score = metrics.get("ood_test_acc", float("nan"))
    return -1.0 if np.isnan(score) else float(score)


def run_method_with_cfg(args, method: str, data, masks, pre_ckpt, train_cfg: Dict[str, object], verbose: bool):
    model = build_model_from_checkpoint(pre_ckpt, data.x.device)
    model_cfg = pre_ckpt["model_cfg"]
    run_fn = METHOD_SPECS[method]["run"]
    return run_fn(
        args,
        model,
        data,
        masks,
        train_cfg,
        model_cfg,
        verbose,
    )


def default_train_cfg(args, method: str, num_layers: int) -> Dict[str, object]:
    cfg_fn = METHOD_SPECS[method]["train_cfg"]
    return cfg_fn(args, num_layers)


def build_search_space(method: str, trial, args, num_layers: int) -> Dict[str, object]:
    search_fn = METHOD_SPECS[method]["search_space"]
    return search_fn(trial, args, num_layers)


def run_one_method(args, method: str, data, masks, pre_ckpt, base_out_dir: Path):
    num_layers = int(pre_ckpt["model_cfg"]["num_layers"])

    if method == "stage1" or not args.use_optuna:
        best_train_cfg = default_train_cfg(args, method, num_layers)
        metrics, artifacts = run_method_with_cfg(
            args=args,
            method=method,
            data=data,
            masks=masks,
            pre_ckpt=pre_ckpt,
            train_cfg=best_train_cfg,
            verbose=True,
        )
    else:
        if optuna is None:
            raise RuntimeError("Optuna is not installed, but --use-optuna is set")

        best_trial_score = float("-inf")
        best_trial_metrics = None
        best_trial_artifacts = None
        best_train_cfg = None

        def objective(trial):
            nonlocal best_trial_score, best_trial_metrics, best_trial_artifacts, best_train_cfg
            set_seed(int(args.seed) + int(trial.number))
            train_cfg = build_search_space(method, trial, args, num_layers)
            metrics_tmp, artifacts_tmp = run_method_with_cfg(
                args=args,
                method=method,
                data=data,
                masks=masks,
                pre_ckpt=pre_ckpt,
                train_cfg=train_cfg,
                verbose=False,
            )
            score = select_score(method, metrics_tmp)
            if score > best_trial_score:
                best_trial_score = float(score)
                best_trial_metrics = copy.deepcopy(metrics_tmp)
                best_trial_artifacts = copy.deepcopy(artifacts_tmp)
                best_train_cfg = dict(train_cfg)
            return score

        timeout = None if int(args.optuna_timeout) <= 0 else int(args.optuna_timeout)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=int(args.optuna_trials), timeout=timeout)
        if best_trial_metrics is None or best_trial_artifacts is None or best_train_cfg is None:
            raise RuntimeError("Optuna finished without a valid best trial result for method {}".format(method))
        metrics = best_trial_metrics
        artifacts = best_trial_artifacts

    return save_method_result(args, method, metrics, artifacts, best_train_cfg, base_out_dir)


def save_method_result(
    args,
    method: str,
    metrics: Dict[str, float],
    artifacts: Dict[str, object],
    train_cfg: Dict[str, object],
    base_out_dir: Path,
) -> Dict[str, object]:
    method_dir = base_out_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = method_dir / "baseline_model.pt"
    metrics_path = method_dir / "metrics.json"
    params_path = method_dir / "params.json"

    torch.save(
        {
            "stage": "baseline_compare",
            "method": method,
            "pretrain_ckpt": str(args.pretrain_ckpt),
            "dataset": args.dataset,
            "domain": args.domain,
            "shift": args.shift,
            "task_names": artifacts.get("task_names", []),
            "task_cfg": artifacts.get("task_cfg", {}),
            "model_cfg": artifacts["model_cfg"],
            "train_cfg": train_cfg,
            "metrics": metrics,
            "history": artifacts.get("history", []),
            "state_dict": artifacts["model_state_dict"],
        },
        ckpt_path,
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    run_params = {
        "params_file": args.params_file,
        "method": method,
        "dataset": args.dataset,
        "domain": args.domain,
        "shift": args.shift,
        "pretrain_ckpt": str(args.pretrain_ckpt),
        "train_cfg": train_cfg,
        "use_optuna": bool(args.use_optuna),
        "optuna_trials": int(args.optuna_trials),
        "optuna_timeout": int(args.optuna_timeout),
        "seed": int(args.seed),
        "device": str(args.device),
        "output_root": str(args.output_root),
        "output_dir": str(method_dir),
    }
    run_params.update(train_cfg)
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)

    return {
        "method": method,
        "ood_test_acc": float(metrics["ood_test_acc"]),
        "metrics": metrics,
        "train_cfg": train_cfg,
        "ckpt_path": str(ckpt_path),
        "metrics_path": str(metrics_path),
        "params_path": str(params_path),
    }


def main():
    args = parse_args()
    if not args.pretrain_ckpt:
        raise ValueError("--pretrain-ckpt is required (or provide it via --params-file)")

    set_seed(int(args.seed))
    device = torch.device(args.device)

    pre_ckpt_path = Path(args.pretrain_ckpt)
    pre_ckpt = safe_torch_load(pre_ckpt_path)
    pre_ckpt["model_cfg"] = normalize_pretrain_model_cfg(pre_ckpt, pre_ckpt_path)

    dataset = args.dataset or pre_ckpt["dataset"]
    domain = args.domain or pre_ckpt["domain"]
    shift = args.shift or pre_ckpt["shift"]
    args.dataset = dataset
    args.domain = domain
    args.shift = shift

    data, masks, data_path = load_good_data(dataset, domain, shift, device)

    ts = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d-%H%M%S")
    base_out_dir = Path(args.output_root) / "baseline_compare" / dataset.lower() / domain.lower() / shift.lower() / ts
    base_out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    methods = resolve_methods(args)
    for method in methods:
        print("\n========== Baseline Method: {} ==========".format(method))
        result = run_one_method(
            args=args,
            method=method,
            data=data,
            masks=masks,
            pre_ckpt=pre_ckpt,
            base_out_dir=base_out_dir,
        )
        results.append(result)
        print("[{}] ood_test_acc={:.6f}".format(method, float(result["ood_test_acc"])))

    best = max(results, key=lambda item: float(item["ood_test_acc"]))
    summary = {
        "run_name": str(args.run_name),
        "dataset": dataset,
        "domain": domain,
        "shift": shift,
        "data_path": str(data_path),
        "pretrain_ckpt": str(args.pretrain_ckpt),
        "methods": methods,
        "best_method": best["method"],
        "best_ood_test_acc": float(best["ood_test_acc"]),
        "results": results,
    }

    summary_path = base_out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved baseline summary:", summary_path)
    print("Best method:", best["method"])
    print("Best ood_test_acc:", float(best["ood_test_acc"]))
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
