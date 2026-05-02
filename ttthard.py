from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from ssl_tasks import build_ssl_tasks
from ttt import (
    GNNNodeClassifier,
    GateMLP,
    accuracy,
    build_ttt_search_space,
    clamp_replace_last_k_layers,
    load_flat_params,
    load_good_data,
    make_stage_output_dir,
    mixed_encoder_embed,
    normalize_pretrain_model_cfg,
    safe_torch_load,
    save_ttt_curve_plot,
    set_seed,
)

try:
    import optuna
except Exception:
    optuna = None


class GateRangeAdapter(nn.Module):
    """Map each gate dimension to [0,1] by min-max, then renormalize to simplex."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("fitted_min", torch.empty(0))
        self.register_buffer("fitted_max", torch.empty(0))

    @torch.no_grad()
    def fit(self, weights: torch.Tensor) -> None:
        self.fitted_min = weights.min(dim=0).values
        self.fitted_max = weights.max(dim=0).values

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        if self.fitted_min.numel() == 0 or self.fitted_max.numel() == 0:
            raise RuntimeError("GateRangeAdapter must be fit() before forward().")
        w = (weights - self.fitted_min) / (self.fitted_max - self.fitted_min + self.eps)
        w = w.clamp(0.0, 1.0)
        return w / w.sum(dim=-1, keepdim=True).clamp_min(self.eps)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-3 hard TTT with gate range adapter")
    parser.add_argument("--pretrain-ckpt", type=str, default="")
    parser.add_argument("--gate-ckpt", type=str, default="")

    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--shift", type=str, default="")

    parser.add_argument("--ssl-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--replace-last-k-layers", type=int, default=1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-optuna", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--optuna-timeout", type=int, default=0)

    parser.add_argument("--adapter-eps", type=float, default=1e-8)

    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument("--run-name", type=str, default="ttthard")
    parser.add_argument("--params-file", type=str, default="")

    prelim = parser.parse_known_args()[0]
    if prelim.params_file:
        parser.set_defaults(**load_flat_params(prelim.params_file))
    return parser.parse_args()


def run_ttthard_once(args, data, masks, pre_ckpt, gate_ckpt, train_cfg: Dict[str, float], verbose: bool = True):
    model_cfg = pre_ckpt["model_cfg"]
    ood_test_mask = masks["ood_test"]

    pre_model = GNNNodeClassifier(
        in_dim=int(model_cfg["in_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        gnn_type=str(model_cfg["gnn_type"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        use_bn=bool(model_cfg.get("use_bn", True)),
        classifier_layers=int(model_cfg.get("classifier_layers", 1)),
    ).to(data.x.device)
    pre_model.load_state_dict(pre_ckpt["state_dict"])
    pre_model.eval()

    gate_cfg = gate_ckpt["gate_cfg"]
    gate = GateMLP(
        in_dim=int(gate_cfg["in_dim"]),
        hidden_dim=int(gate_cfg["hidden_dim"]),
        num_layers=int(gate_cfg["num_layers"]),
        out_dim=int(gate_cfg["out_dim"]),
        dropout=float(gate_cfg["dropout"]),
        temperature=float(gate_cfg.get("temperature", 1.0)),
    ).to(data.x.device)
    gate.load_state_dict(gate_ckpt["gate_state_dict"])
    gate.eval()
    for p in gate.parameters():
        p.requires_grad = False

    task_names = gate_ckpt["task_names"]
    task_cfg = gate_ckpt.get("task_cfg", {})
    replace_last_k_layers = clamp_replace_last_k_layers(
        int(train_cfg.get("replace_last_k_layers", 1)),
        int(model_cfg["num_layers"]),
    )

    adapted_models = []
    for _ in task_names:
        model_i = GNNNodeClassifier(
            in_dim=int(model_cfg["in_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            out_dim=int(model_cfg["out_dim"]),
            gnn_type=str(model_cfg["gnn_type"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            use_bn=bool(model_cfg.get("use_bn", True)),
            classifier_layers=int(model_cfg.get("classifier_layers", 1)),
        ).to(data.x.device)
        model_i.load_state_dict(pre_ckpt["state_dict"])
        model_i.eval()
        adapted_models.append(model_i)

    with torch.no_grad():
        pre_embed = pre_model.get_embed(data.x, data.edge_index)
        base_gate_weights = gate(pre_embed)

    adapter = GateRangeAdapter(eps=float(args.adapter_eps)).to(data.x.device)
    adapter.fit(base_gate_weights)

    def evaluate_current(models) -> Dict[str, float]:
        with torch.no_grad():
            gate_weights = adapter(gate(pre_embed))
            branch_logits = []
            for branch_model in models:
                h_i = mixed_encoder_embed(
                    data.x,
                    data.edge_index,
                    pre_model=pre_model,
                    adapted_model=branch_model,
                    replace_last_k_layers=replace_last_k_layers,
                )
                branch_logits.append(pre_model.classifier(h_i))
            logits_stack = torch.stack(branch_logits, dim=1)
            fused_logits = (logits_stack * gate_weights.unsqueeze(-1)).sum(dim=1)

        return {
            "id_val_acc": accuracy(fused_logits, data.y, masks["id_val"]),
            "id_test_acc": accuracy(fused_logits, data.y, masks["id_test"]),
            "ood_val_acc": accuracy(fused_logits, data.y, masks["ood_val"]),
            "ood_test_acc": accuracy(fused_logits, data.y, masks["ood_test"]),
        }

    best_ood_test = float("-inf")
    best_metrics = None
    best_model_states = [copy.deepcopy(model.state_dict()) for model in adapted_models]
    history = []

    def record_history(epoch_idx: int, metrics_dict: Dict[str, float]) -> None:
        history.append(
            {
                "epoch": int(epoch_idx),
                "ood_test_acc": float(metrics_dict["ood_test_acc"]),
                "id_test_acc": float(metrics_dict["id_test_acc"]),
                "ood_val_acc": float(metrics_dict["ood_val_acc"]),
                "id_val_acc": float(metrics_dict["id_val_acc"]),
            }
        )

    initial_metrics = evaluate_current(adapted_models)
    record_history(0, initial_metrics)
    if not np.isnan(initial_metrics["ood_test_acc"]):
        best_ood_test = float(initial_metrics["ood_test_acc"])
        best_metrics = dict(initial_metrics)

    branch_ssl_tasks = []
    branch_optimizers = []
    for idx, task_name in enumerate(task_names):
        model_i = adapted_models[idx]
        ssl_i = build_ssl_tasks(
            [task_name],
            task_cfg=task_cfg,
            device=data.x.device,
            hidden_dim=int(model_cfg["hidden_dim"]),
            input_dim=int(model_cfg["in_dim"]),
        )
        ssl_params = list(ssl_i.parameters())
        for p in model_i.convs.parameters():
            p.requires_grad = True

        param_groups = [{"params": model_i.convs.parameters(), "lr": float(train_cfg["encoder_lr"])}]
        if len(ssl_params) > 0:
            param_groups.append({"params": ssl_params, "lr": float(train_cfg["ssl_lr"])})
        opt_i = torch.optim.Adam(param_groups, weight_decay=float(train_cfg["weight_decay"]))
        branch_ssl_tasks.append(ssl_i)
        branch_optimizers.append(opt_i)

    for epoch in range(1, int(train_cfg["finetune_epochs"]) + 1):
        for idx in range(len(task_names)):
            model_i = adapted_models[idx]
            ssl_i = branch_ssl_tasks[idx]
            opt_i = branch_optimizers[idx]

            model_i.train()
            ssl_i.train()
            opt_i.zero_grad()
            node_loss = ssl_i[0].compute_node_loss(model_i, data.x, data.edge_index)
            loss = node_loss[ood_test_mask].mean()
            if loss.requires_grad:
                loss.backward()
                opt_i.step()

        current_metrics = evaluate_current(adapted_models)
        record_history(epoch, current_metrics)
        if not np.isnan(current_metrics["ood_test_acc"]) and current_metrics["ood_test_acc"] > best_ood_test:
            best_ood_test = float(current_metrics["ood_test_acc"])
            best_metrics = dict(current_metrics)
            best_model_states = [copy.deepcopy(model.state_dict()) for model in adapted_models]

    for model_i, state in zip(adapted_models, best_model_states):
        model_i.load_state_dict(state)

    with torch.no_grad():
        adapted_gate_weights = adapter(gate(pre_embed))

    if verbose:
        print("Adapted branches: {}".format(", ".join("{}:{}".format(i + 1, n) for i, n in enumerate(task_names))))
        print("[ttthard] gate weight sum mean={:.6f}".format(float(adapted_gate_weights.sum(dim=-1).mean().item())))

    artifacts = {
        "model_cfg": model_cfg,
        "gate_cfg": gate_cfg,
        "gate_state_dict": gate.state_dict(),
        "classifier_state_dict": pre_model.classifier.state_dict(),
        "adapted_model_state_dicts": [model_i.state_dict() for model_i in adapted_models],
        "task_names": task_names,
        "task_cfg": task_cfg,
        "replace_last_k_layers": int(replace_last_k_layers),
        "history": history,
        "adapter_cfg": {"type": "minmax_simplex", "eps": float(args.adapter_eps)},
        "gate_weight_stats": {
            "raw_min": base_gate_weights.min(dim=0).values.detach().cpu().tolist(),
            "raw_max": base_gate_weights.max(dim=0).values.detach().cpu().tolist(),
            "adapted_min": adapted_gate_weights.min(dim=0).values.detach().cpu().tolist(),
            "adapted_max": adapted_gate_weights.max(dim=0).values.detach().cpu().tolist(),
            "sum_mean": float(adapted_gate_weights.sum(dim=-1).mean().item()),
        },
    }
    return best_metrics if best_metrics is not None else initial_metrics, artifacts


def main():
    args = parse_args()
    if not args.pretrain_ckpt or not args.gate_ckpt:
        raise ValueError("--pretrain-ckpt and --gate-ckpt are required (or provide them via --params-file)")

    set_seed(int(args.seed))
    device = torch.device(args.device)

    pre_ckpt_path = Path(args.pretrain_ckpt)
    pre_ckpt = safe_torch_load(pre_ckpt_path)
    gate_ckpt = safe_torch_load(Path(args.gate_ckpt))
    pre_ckpt["model_cfg"] = normalize_pretrain_model_cfg(pre_ckpt, pre_ckpt_path)

    model_cfg = pre_ckpt["model_cfg"]
    dataset = args.dataset or gate_ckpt.get("dataset", pre_ckpt["dataset"])
    domain = args.domain or gate_ckpt.get("domain", pre_ckpt["domain"])
    shift = args.shift or gate_ckpt.get("shift", pre_ckpt["shift"])
    data, masks, data_path = load_good_data(dataset, domain, shift, device)

    if args.use_optuna:
        if optuna is None:
            raise RuntimeError("Optuna is not installed, but --use-optuna is set")

        best_trial_score = float("-inf")
        best_trial_metrics = None
        best_trial_artifacts = None
        best_train_cfg = None

        def objective(trial):
            nonlocal best_trial_score, best_trial_metrics, best_trial_artifacts, best_train_cfg
            train_cfg = build_ttt_search_space(trial, args, int(model_cfg["num_layers"]))
            metrics_tmp, artifacts_tmp = run_ttthard_once(
                args=args,
                data=data,
                masks=masks,
                pre_ckpt=pre_ckpt,
                gate_ckpt=gate_ckpt,
                train_cfg=train_cfg,
                verbose=False,
            )
            score = metrics_tmp["ood_test_acc"]
            if np.isnan(score):
                score = -1.0
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
            raise RuntimeError("Optuna finished without a valid best trial result to save.")
        metrics = best_trial_metrics
        artifacts = best_trial_artifacts
    else:
        best_train_cfg = {
            "ssl_lr": float(args.ssl_lr),
            "encoder_lr": float(args.encoder_lr),
            "weight_decay": float(args.weight_decay),
            "finetune_epochs": int(args.finetune_epochs),
            "replace_last_k_layers": clamp_replace_last_k_layers(
                int(args.replace_last_k_layers),
                int(model_cfg["num_layers"]),
            ),
        }
        metrics, artifacts = run_ttthard_once(
            args=args,
            data=data,
            masks=masks,
            pre_ckpt=pre_ckpt,
            gate_ckpt=gate_ckpt,
            train_cfg=best_train_cfg,
            verbose=True,
        )

    out_dir = make_stage_output_dir(
        args.output_root,
        "stage3_hard",
        dataset,
        domain,
        shift,
        timestamp=args.timestamp,
    )
    ckpt_path = out_dir / "ttthard_model.pt"
    metrics_path = out_dir / "metrics.json"
    params_path = out_dir / "params.json"
    plot_path = out_dir / "ood_test_acc_vs_epoch.png"

    save_ttt_curve_plot(artifacts.get("history", []), plot_path)

    torch.save(
        {
            "stage": "ttthard",
            "pretrain_ckpt": str(args.pretrain_ckpt),
            "gate_ckpt": str(args.gate_ckpt),
            "dataset": dataset,
            "domain": domain,
            "shift": shift,
            "data_path": str(data_path),
            "task_names": artifacts["task_names"],
            "task_cfg": artifacts["task_cfg"],
            "replace_last_k_layers": int(artifacts.get("replace_last_k_layers", 1)),
            "model_cfg": artifacts["model_cfg"],
            "gate_cfg": artifacts["gate_cfg"],
            "gate_state_dict": artifacts["gate_state_dict"],
            "classifier_state_dict": artifacts["classifier_state_dict"],
            "adapted_model_state_dicts": artifacts["adapted_model_state_dicts"],
            "adapter_cfg": artifacts.get("adapter_cfg", {}),
            "gate_weight_stats": artifacts.get("gate_weight_stats", {}),
            "train_cfg": best_train_cfg,
            "metrics": metrics,
            "history": artifacts.get("history", []),
        },
        ckpt_path,
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    run_params = {
        "params_file": args.params_file,
        "dataset": dataset,
        "domain": domain,
        "shift": shift,
        "pretrain_ckpt": str(args.pretrain_ckpt),
        "gate_ckpt": str(args.gate_ckpt),
        "ssl_lr": float(best_train_cfg["ssl_lr"]),
        "encoder_lr": float(best_train_cfg["encoder_lr"]),
        "weight_decay": float(best_train_cfg["weight_decay"]),
        "finetune_epochs": int(best_train_cfg["finetune_epochs"]),
        "replace_last_k_layers": int(best_train_cfg["replace_last_k_layers"]),
        "adapter_eps": float(args.adapter_eps),
        "use_optuna": bool(args.use_optuna),
        "optuna_trials": int(args.optuna_trials),
        "optuna_timeout": int(args.optuna_timeout),
        "seed": int(args.seed),
        "device": str(args.device),
        "output_root": str(args.output_root),
        "timestamp": str(args.timestamp),
        "run_name": str(args.run_name),
        "output_dir": str(out_dir),
        "plot_path": str(plot_path),
    }
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)

    print("Saved ttthard checkpoint:", ckpt_path)
    print("Saved params:", params_path)
    print("Saved plot:", plot_path)
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
