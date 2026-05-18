from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from gate_ssl import (
    FrozenEmbeddingModel,
    GNNNodeClassifier,
    GateMLP,
    available_ssl_tasks,
    build_ssl_tasks,
    compute_task_loss_matrix,
    load_flat_params,
    load_good_data,
    make_ssl_tag,
    make_stage_output_dir,
    normalize_pretrain_model_cfg,
    parse_ssl_task_names,
    safe_torch_load,
    set_seed,
)


def build_inverse_relative_target(loss_mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    node_sum = loss_mat.sum(dim=-1, keepdim=True)
    rel_loss = loss_mat / node_sum.clamp_min(eps)
    inv_rel = rel_loss.clamp_min(eps).reciprocal()
    return inv_rel / inv_rel.sum(dim=-1, keepdim=True).clamp_min(eps)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 train SSL heads + gate with KL target")
    parser.add_argument("--pretrain-ckpt", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--shift", type=str, default="")

    parser.add_argument("--ssl-tasks", type=str, default="consistency")
    parser.add_argument("--num-ssl", type=int, default=1)
    parser.add_argument("--task-cfg-json", type=str, default="")

    parser.add_argument("--gate-hidden-dim", type=int, default=128)
    parser.add_argument("--gate-num-layers", type=int, default=2)
    parser.add_argument("--gate-dropout", type=float, default=0.1)
    parser.add_argument("--gate-temperature", type=float, default=1.0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument("--run-name", type=str, default="gate_ssl_kl")
    parser.add_argument("--params-file", type=str, default="")

    prelim = parser.parse_known_args()[0]
    if prelim.params_file:
        parser.set_defaults(**load_flat_params(prelim.params_file))
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.pretrain_ckpt:
        raise ValueError("--pretrain-ckpt is required (or provide it via --params-file)")

    set_seed(int(args.seed))
    device = torch.device(args.device)

    pretrain_ckpt_path = Path(args.pretrain_ckpt)
    pretrain_ckpt = safe_torch_load(pretrain_ckpt_path)
    model_cfg = normalize_pretrain_model_cfg(pretrain_ckpt, pretrain_ckpt_path)

    dataset = args.dataset or pretrain_ckpt["dataset"]
    domain = args.domain or pretrain_ckpt["domain"]
    shift = args.shift or pretrain_ckpt["shift"]

    data, masks, data_path = load_good_data(dataset, domain, shift, device)

    base_model = GNNNodeClassifier(
        in_dim=int(model_cfg["in_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        gnn_type=str(model_cfg["gnn_type"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        use_bn=bool(model_cfg.get("use_bn", True)),
        classifier_layers=int(model_cfg.get("classifier_layers", 1)),
    ).to(device)
    base_model.load_state_dict(pretrain_ckpt["state_dict"])
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        base_embed = base_model.get_embed(data.x, data.edge_index)

    frozen_model = FrozenEmbeddingModel(base_embed, base_model.classifier).to(device)

    if args.ssl_tasks.lower() == "all":
        task_names = available_ssl_tasks()
    else:
        task_names = parse_ssl_task_names(args.ssl_tasks)
    if int(args.num_ssl) > 0:
        if int(args.num_ssl) > len(task_names):
            raise ValueError("num_ssl={} is larger than parsed task count={}".format(args.num_ssl, len(task_names)))
        task_names = task_names[: int(args.num_ssl)]
    if not task_names:
        raise ValueError("No SSL tasks selected")

    task_cfg = json.loads(args.task_cfg_json) if args.task_cfg_json else {}
    ssl_tasks = build_ssl_tasks(
        task_names,
        task_cfg=task_cfg,
        device=device,
        hidden_dim=int(model_cfg["hidden_dim"]),
        input_dim=int(model_cfg["in_dim"]),
    )

    gate = GateMLP(
        in_dim=int(model_cfg["hidden_dim"]),
        hidden_dim=int(args.gate_hidden_dim),
        num_layers=int(args.gate_num_layers),
        out_dim=len(task_names),
        dropout=float(args.gate_dropout),
        temperature=float(args.gate_temperature),
    ).to(device)

    ssl_param_list = list(ssl_tasks.parameters())
    ssl_optimizer = None
    if len(ssl_param_list) > 0:
        ssl_optimizer = torch.optim.Adam(
            ssl_param_list,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
    gate_optimizer = torch.optim.Adam(
        gate.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_val = float("inf")
    best_state = None
    wait = 0
    train_mask = masks["train"]
    ssl_skipped_no_grad = 0

    for _ in range(1, int(args.epochs) + 1):
        gate.train()
        ssl_tasks.train()

        if ssl_optimizer is not None:
            ssl_optimizer.zero_grad()
        gate_optimizer.zero_grad()

        loss_mat = compute_task_loss_matrix(
            ssl_tasks,
            frozen_model,
            feat=data.x,
            edge_index=data.edge_index,
        )
        ssl_loss = loss_mat[train_mask].mean()
        gate_target = build_inverse_relative_target(loss_mat.detach())
        gate_weights = gate(base_embed)
        gate_log_prob = gate_weights.clamp_min(1e-12).log()
        gate_kl_per_node = F.kl_div(gate_log_prob, gate_target, reduction="none").sum(dim=-1)
        gate_loss = gate_kl_per_node[train_mask].mean()

        if ssl_optimizer is not None and ssl_loss.requires_grad:
            ssl_loss.backward()
            ssl_optimizer.step()
        elif ssl_optimizer is not None:
            ssl_skipped_no_grad += 1

        gate_loss.backward()
        gate_optimizer.step()

        with torch.no_grad():
            gate.eval()
            ssl_tasks.eval()
            val_mask = masks["id_val"] if masks["id_val"] is not None else train_mask
            val_loss_mat = compute_task_loss_matrix(
                ssl_tasks,
                frozen_model,
                feat=data.x,
                edge_index=data.edge_index,
            )
            val_ssl_loss = val_loss_mat[val_mask].mean().item()
            val_target = build_inverse_relative_target(val_loss_mat)
            val_gate_weights = gate(base_embed)
            val_gate_kl = (
                F.kl_div(val_gate_weights.clamp_min(1e-12).log(), val_target, reduction="none")
                .sum(dim=-1)[val_mask]
                .mean()
                .item()
            )
            val_metric = val_ssl_loss + val_gate_kl

        if val_metric < best_val:
            best_val = val_metric
            best_state = {
                "gate": gate.state_dict(),
                "ssl": [task.state_dict() for task in ssl_tasks],
            }
            wait = 0
        else:
            wait += 1
            if wait >= int(args.patience):
                break

    if best_state is not None:
        gate.load_state_dict(best_state["gate"])
        for task, state in zip(ssl_tasks, best_state["ssl"]):
            task.load_state_dict(state)

    if ssl_skipped_no_grad > 0:
        print("[stage2] Skip backward when SSL loss has no grad graph: ssl_skipped={}".format(ssl_skipped_no_grad))

    gate.eval()
    ssl_tasks.eval()
    with torch.no_grad():
        final_loss_mat = compute_task_loss_matrix(
            ssl_tasks,
            frozen_model,
            feat=data.x,
            edge_index=data.edge_index,
        )
        final_target = build_inverse_relative_target(final_loss_mat)
        final_weights = gate(base_embed)
        final_gate_kl = F.kl_div(final_weights.clamp_min(1e-12).log(), final_target, reduction="none").sum(dim=-1)

    out_dir = make_stage_output_dir(
        args.output_root,
        "stage2",
        dataset,
        domain,
        shift,
        make_ssl_tag(task_names),
        timestamp=args.timestamp,
    )
    ckpt_path = out_dir / "gate_model.pt"
    plot_path = out_dir / "plot.png"
    metrics_path = out_dir / "metrics.json"
    params_path = out_dir / "params.json"

    torch.save(
        {
            "stage": "gate_ssl_kl",
            "pretrain_ckpt": str(args.pretrain_ckpt),
            "dataset": dataset,
            "domain": domain,
            "shift": shift,
            "data_path": str(data_path),
            "model_cfg": model_cfg,
            "task_names": task_names,
            "task_cfg": task_cfg,
            "gate_cfg": {
                "in_dim": int(model_cfg["hidden_dim"]),
                "hidden_dim": int(args.gate_hidden_dim),
                "num_layers": int(args.gate_num_layers),
                "dropout": float(args.gate_dropout),
                "temperature": float(args.gate_temperature),
                "out_dim": len(task_names),
            },
            "loss_mode": "ssl_total_plus_gate_kl",
            "gate_target_mode": "inverse_relative_ssl_loss",
            "gate_state_dict": gate.state_dict(),
        },
        ckpt_path,
    )

    if len(task_names) == 1:
        y = final_loss_mat[:, 0].detach().cpu().numpy()
        x = np.arange(y.shape[0])
        plt.figure(figsize=(8, 4))
        plt.scatter(x, y, s=6, alpha=0.8)
        plt.xlim(0, max(1, y.shape[0]))
        plt.xlabel("Node Index")
        plt.ylabel("SSL Loss")
        plt.title("{} node-wise SSL loss".format(task_names[0]))
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()
    else:
        x = final_target[:, 0].detach().cpu().numpy()
        y = final_weights[:, 0].detach().cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.scatter(x, y, s=6, alpha=0.7)
        plt.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xlabel("Target weight: {}".format(task_names[0]))
        plt.ylabel("Gate weight: {}".format(task_names[0]))
        plt.title("Gate KL alignment")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()

    summary = {
        "task_names": task_names,
        "mean_gate_weight": final_weights.mean(dim=0).detach().cpu().tolist(),
        "mean_target_weight": final_target.mean(dim=0).detach().cpu().tolist(),
        "mean_gate_kl": float(final_gate_kl.mean().item()),
        "mean_ssl_loss": float(final_loss_mat.mean().item()),
        "plot_path": str(plot_path),
        "ckpt_path": str(ckpt_path),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    run_params = {
        "params_file": args.params_file,
        "dataset": dataset,
        "domain": domain,
        "shift": shift,
        "pretrain_ckpt": str(args.pretrain_ckpt),
        "ssl_tasks": ",".join(task_names),
        "num_ssl": int(len(task_names)),
        "task_cfg_json": json.dumps(task_cfg, ensure_ascii=True),
        "gate_hidden_dim": int(args.gate_hidden_dim),
        "gate_num_layers": int(args.gate_num_layers),
        "gate_dropout": float(args.gate_dropout),
        "gate_temperature": float(args.gate_temperature),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": str(args.device),
        "output_root": str(args.output_root),
        "timestamp": str(args.timestamp),
        "run_name": str(args.run_name),
        "loss_mode": "ssl_total_plus_gate_kl",
        "gate_target_mode": "inverse_relative_ssl_loss",
        "output_dir": str(out_dir),
    }
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)

    print("Saved checkpoint:", ckpt_path)
    print("Saved params:", params_path)
    print("Saved plot:", plot_path)
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
