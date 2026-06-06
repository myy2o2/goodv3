from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from ssl_tasks import available_ssl_tasks, build_ssl_tasks, parse_ssl_task_names
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
    save_gate_weight_plot,
    save_ttt_curve_plot,
    set_seed,
)

try:
    import optuna
except Exception:
    optuna = None


# This variant keeps the original stage2/stage3 parameter-reading style,
# but replaces the stage2-trained gate with a freshly initialized gate that
# can optionally keep updating during stage3.


def parse_task_cfg_json(task_cfg_json: str) -> Dict[str, object]:
    # 功能概括：把 stage2 风格传入的 task_cfg_json 字符串解析成 Python 字典。
    if not task_cfg_json:
        # 没传时沿用空配置；各个 SSL 任务会使用自己的默认参数。
        return {}
    # stage2 原本就是把这类字段当成 JSON 字符串来读，这里保持同样约定。
    task_cfg = json.loads(task_cfg_json)
    if not isinstance(task_cfg, dict):
        raise ValueError("task_cfg_json must decode to a JSON object")
    return task_cfg


def resolve_stage2_task_names(ssl_tasks: str, num_ssl: int) -> List[str]:
    # 功能概括：按 stage2 的规则解析任务名，并应用 num_ssl 的截断语义。
    if ssl_tasks.lower() == "all":
        # all 表示启用仓库里当前所有可用的 SSL 任务。
        task_names = available_ssl_tasks()
    else:
        # 非 all 时，按逗号分隔等规则解析具体任务名。
        task_names = parse_ssl_task_names(ssl_tasks)
    if int(num_ssl) > 0:
        if int(num_ssl) > len(task_names):
            raise ValueError("num_ssl={} is larger than parsed task count={}".format(num_ssl, len(task_names)))
        # stage2 原本会只取前 num_ssl 个任务，这里完全保持一致。
        task_names = task_names[: int(num_ssl)]
    if not task_names:
        raise ValueError("No SSL tasks selected")
    return task_names


def init_gate_parameters(gate: nn.Module, mode: str, scale: float) -> None:
    # 功能概括：给 gate 做初始化；这是这个脚本相对原版 ttt_shared 的关键新增点之一。
    gain = max(float(scale), 1e-8)
    init_mode = str(mode).strip().lower()
    with torch.no_grad():
        # 逐个参数张量处理，避免在初始化时留下梯度记录。
        for name, param in gate.named_parameters():
            if param.ndim <= 1:
                # 一维张量一般是 bias 或 norm 向量。
                # bias 初始化为 0，其他一维参数初始化为 1，保证起点稳定。
                if "bias" in name:
                    param.zero_()
                else:
                    param.fill_(1.0)
                continue

            # 二维及以上张量一般是线性层权重，根据用户指定的模式初始化。
            if init_mode == "xavier":
                nn.init.xavier_uniform_(param, gain=gain)
            elif init_mode == "kaiming":
                nn.init.kaiming_uniform_(param, a=np.sqrt(5.0))
                param.mul_(gain)
            elif init_mode == "normal":
                nn.init.normal_(param, mean=0.0, std=gain)
            elif init_mode == "uniform":
                nn.init.uniform_(param, a=-gain, b=gain)
            elif init_mode == "near_uniform":
                nn.init.normal_(param, mean=0.0, std=gain * 1e-2)
            elif init_mode == "zeros":
                param.zero_()
            else:
                raise ValueError("Unsupported --gate-init mode: {}".format(mode))


def build_gate_cfg(args, model_cfg: Dict[str, object], task_count: int) -> Dict[str, object]:
    # 功能概括：把 stage2 风格的 gate 超参数整理成一个统一配置字典。
    return build_gate_cfg_with_temperature(args, model_cfg, task_count, float(args.gate_temperature))


def build_gate_cfg_with_temperature(
    args,
    model_cfg: Dict[str, object],
    task_count: int,
    gate_temperature: float,
) -> Dict[str, object]:
    return {
        # gate 输入维度固定取 encoder 的 hidden_dim，这和原 stage2 保持一致。
        "in_dim": int(model_cfg["hidden_dim"]),
        "hidden_dim": int(args.gate_hidden_dim),
        "num_layers": int(args.gate_num_layers),
        # 输出维度就是任务数，因为 gate 要给每个任务分一个权重。
        "out_dim": int(task_count),
        "dropout": float(args.gate_dropout),
        "temperature": float(gate_temperature),
        # 下面三项是这个脚本新增的 gate 行为控制项。
        "init": str(args.gate_init),
        "init_scale": float(args.gate_init_scale),
        "freeze_gate": bool(args.freeze_gate),
    }


def build_base_train_cfg(args, model_cfg: Dict[str, object]) -> Dict[str, float]:
    # 功能概括：整理 stage3 常规训练时使用的训练超参数。
    return {
        "ssl_lr": float(args.ssl_lr),
        "encoder_lr": float(args.encoder_lr),
        # gate_lr 是本脚本新增的，因为 gate 现在可以选择参与更新。
        "gate_lr": float(args.gate_lr),
        "weight_decay": float(args.weight_decay),
        "finetune_epochs": int(args.finetune_epochs),
        # replace_last_k_layers 仍然沿用原版 stage3 的约束逻辑。
        "replace_last_k_layers": clamp_replace_last_k_layers(
            int(args.replace_last_k_layers),
            int(model_cfg["num_layers"]),
        ),
    }


def build_trial_train_cfg(trial, args, model_cfg: Dict[str, object]) -> Dict[str, float]:
    # 功能概括：整理 Optuna trial 对应的训练超参数。
    # 除了 gate_lr 之外，其余被搜索的训练字段完全复用原版 stage3 的搜索空间定义。
    train_cfg = build_ttt_search_space(trial, args, int(model_cfg["num_layers"]))
    train_cfg.update(
        {
            "gate_lr": float(args.gate_lr),
        }
    )
    return train_cfg


def build_trial_gate_cfg(trial, args, model_cfg: Dict[str, object], task_count: int) -> Dict[str, object]:
    # 功能概括：整理 Optuna trial 对应的 gate 超参数。
    gate_temperature = trial.suggest_float(
        "gate_temperature",
        float(args.gate_temperature_min),
        float(args.gate_temperature_max),
        log=True,
    )
    return build_gate_cfg_with_temperature(args, model_cfg, task_count, gate_temperature)


def parse_args():
    # 功能概括：定义命令行参数入口。
    # 这里刻意把参数分成两组：
    # 1) stage2 风格：任务选择、task_cfg、gate 结构
    # 2) stage3 风格：TTT 训练和搜参参数
    parser = argparse.ArgumentParser(
        description="One-stage shared-encoder TTT with freshly initialized gate"
    )
    # stage1 checkpoint 是整个脚本唯一必须提供的输入模型。
    parser.add_argument("--pretrain-ckpt", type=str, default="")

    # 数据集三元组默认可从 pretrain checkpoint 回填。
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--shift", type=str, default="")

    # Stage2-style task and gate configuration.
    parser.add_argument("--ssl-tasks", type=str, default="consistency")
    parser.add_argument("--num-ssl", type=int, default=1)
    parser.add_argument("--task-cfg-json", type=str, default="")
    parser.add_argument("--gate-hidden-dim", type=int, default=128)
    parser.add_argument("--gate-num-layers", type=int, default=2)
    parser.add_argument("--gate-dropout", type=float, default=0.1)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--gate-temperature-min", type=float, default=0.5)
    parser.add_argument("--gate-temperature-max", type=float, default=10.0)
    parser.add_argument(
        "--gate-init",
        type=str,
        default="near_uniform",
        choices=["near_uniform", "xavier", "kaiming", "normal", "uniform", "zeros"],
    )
    parser.add_argument("--gate-init-scale", type=float, default=1.0)
    parser.add_argument("--freeze-gate", action="store_true")

    # Stage3-style TTT configuration.
    parser.add_argument("--ssl-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=5e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--replace-last-k-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-optuna", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--optuna-timeout", type=int, default=0)

    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument("--run-name", type=str, default="ttt_shared_onestage")
    parser.add_argument("--params-file", type=str, default="")

    # 和原版脚本一样：如果传了 params-file，就先把其中的字段作为默认值载入。
    prelim = parser.parse_known_args()[0]
    if prelim.params_file:
        parser.set_defaults(**load_flat_params(prelim.params_file))
    return parser.parse_args()


def run_ttt_shared_onestage_once(
    args,
    data,
    masks,
    pre_ckpt,
    task_names: List[str],
    task_cfg: Dict[str, object],
    gate_cfg: Dict[str, object],
    train_cfg: Dict[str, float],
    verbose: bool = True,
):
    # 功能概括：执行一次完整训练/评估。
    # 输入是已经解析好的数据、pretrain checkpoint、任务配置、gate 配置、训练配置。
    # 输出是这次运行的最优指标和保存 checkpoint 所需的 artifacts。
    model_cfg = pre_ckpt["model_cfg"]
    # 这个脚本沿用原版 stage3 shared 的设计：只在 ood_test 节点上做 TTT 损失。
    ood_test_mask = masks["ood_test"]

    # 1) 恢复 stage1 预训练模型；它提供固定前层和最终分类头。
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
    for param in pre_model.parameters():
        # 预训练模型不参与这个阶段的梯度更新。
        param.requires_grad = False

    # 2) 构建 gate，但不再从 stage2 checkpoint 读权重，而是现场初始化。
    gate = GateMLP(
        in_dim=int(gate_cfg["in_dim"]),
        hidden_dim=int(gate_cfg["hidden_dim"]),
        num_layers=int(gate_cfg["num_layers"]),
        out_dim=int(gate_cfg["out_dim"]),
        dropout=float(gate_cfg["dropout"]),
        temperature=float(gate_cfg["temperature"]),
    ).to(data.x.device)
    init_gate_parameters(gate, str(gate_cfg["init"]), float(gate_cfg["init_scale"]))
    for param in gate.parameters():
        # freeze_gate=False 时，gate 会和 encoder、SSL heads 一起更新。
        param.requires_grad = not bool(gate_cfg["freeze_gate"])
    gate.train(not bool(gate_cfg["freeze_gate"]))

    # 3) 决定 mixed encoder 里有多少后层由 adapted_model 替换。
    replace_last_k_layers = clamp_replace_last_k_layers(
        int(train_cfg.get("replace_last_k_layers", 1)),
        int(model_cfg["num_layers"]),
    )

    # 4) 构建可适配分支：初始参数直接复制自 stage1。
    adapted_model = GNNNodeClassifier(
        in_dim=int(model_cfg["in_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        gnn_type=str(model_cfg["gnn_type"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        use_bn=bool(model_cfg.get("use_bn", True)),
        classifier_layers=int(model_cfg.get("classifier_layers", 1)),
    ).to(data.x.device)
    adapted_model.load_state_dict(pre_ckpt["state_dict"])
    adapted_model.eval()

    with torch.no_grad():
        # gate 的输入固定为 pretrain encoder 的输出，这点是你要求保持和旧流程一致的地方。
        pre_embed = pre_model.get_embed(data.x, data.edge_index).detach()

    def evaluate_current(model) -> Dict[str, float]:
        # 功能概括：用当前 adapted_model 做一次完整评估。
        # 这里不会直接拿 adapted_model 全模型输出，而是走 mixed_encoder_embed：
        # 前面几层来自 pre_model，最后 k 层来自 adapted_model，然后接 pre_model.classifier。
        model.eval()
        with torch.no_grad():
            h = mixed_encoder_embed(
                data.x,
                data.edge_index,
                pre_model=pre_model,
                adapted_model=model,
                replace_last_k_layers=replace_last_k_layers,
            )
            # 分类头始终沿用 stage1 的 classifier。
            logits = pre_model.classifier(h)
        return {
            "id_val_acc": accuracy(logits, data.y, masks["id_val"]),
            "id_test_acc": accuracy(logits, data.y, masks["id_test"]),
            "ood_val_acc": accuracy(logits, data.y, masks["ood_val"]),
            "ood_test_acc": accuracy(logits, data.y, masks["ood_test"]),
        }

    # 下面这些变量用于记录训练过程中最好的那一版模型和 gate。
    best_ood_test = float("-inf")
    best_metrics = None
    best_model_state = copy.deepcopy(adapted_model.state_dict())
    best_gate_state = copy.deepcopy(gate.state_dict())
    history = []

    def record_history(epoch_idx: int, metrics_dict: Dict[str, float]) -> None:
        # 功能概括：把每轮的核心指标记下来，后面画曲线和回看训练过程要用。
        history.append(
            {
                "epoch": int(epoch_idx),
                "ood_test_acc": float(metrics_dict["ood_test_acc"]),
                "id_test_acc": float(metrics_dict["id_test_acc"]),
                "ood_val_acc": float(metrics_dict["ood_val_acc"]),
                "id_val_acc": float(metrics_dict["id_val_acc"]),
            }
        )

    # 先记录 epoch 0，也就是还没开始 TTT 适配时的初始表现。
    initial_metrics = evaluate_current(adapted_model)
    record_history(0, initial_metrics)
    if not np.isnan(initial_metrics["ood_test_acc"]) and initial_metrics["ood_test_acc"] > best_ood_test:
        best_ood_test = float(initial_metrics["ood_test_acc"])
        best_metrics = dict(initial_metrics)
        best_model_state = copy.deepcopy(adapted_model.state_dict())
        best_gate_state = copy.deepcopy(gate.state_dict())

    # 5) 构建 SSL 任务模块；任务列表和 task_cfg 的读取方式保持 stage2 原逻辑。
    ssl_tasks = build_ssl_tasks(
        task_names,
        task_cfg=task_cfg,
        device=data.x.device,
        hidden_dim=int(model_cfg["hidden_dim"]),
        input_dim=int(model_cfg["in_dim"]),
    )
    ssl_params = list(ssl_tasks.parameters())

    # 这里只允许适配分支的卷积层更新；分类头仍然固定。
    for param in adapted_model.convs.parameters():
        param.requires_grad = True

    # 参数组拆开是为了给 encoder、SSL task heads、gate 分别设置学习率。
    param_groups = [{"params": adapted_model.convs.parameters(), "lr": float(train_cfg["encoder_lr"])}]
    if len(ssl_params) > 0:
        param_groups.append({"params": ssl_params, "lr": float(train_cfg["ssl_lr"])})
    if not bool(gate_cfg["freeze_gate"]):
        gate_params = [param for param in gate.parameters() if param.requires_grad]
        if gate_params:
            param_groups.append({"params": gate_params, "lr": float(train_cfg["gate_lr"])})

    optimizer = torch.optim.Adam(param_groups, weight_decay=float(train_cfg["weight_decay"]))

    # 6) 主训练循环：每轮都在 OOD test 节点上计算 gate 加权的 SSL 损失。
    for epoch in range(1, int(train_cfg["finetune_epochs"]) + 1):
        adapted_model.train()
        ssl_tasks.train()
        gate.train(not bool(gate_cfg["freeze_gate"]))
        optimizer.zero_grad()

        # gate 只看固定的 pretrain embedding，输出每个节点对每个任务的权重。
        gate_weights = gate(pre_embed)
        if int(gate_weights.shape[-1]) != len(task_names):
            raise ValueError(
                "Gate output dim {} does not match number of task_names {}".format(
                    int(gate_weights.shape[-1]),
                    len(task_names),
                )
            )
        # 只取 OOD test 节点的权重，因为这个阶段的自适应只作用在这些节点上。
        masked_gate_weights = gate_weights[ood_test_mask]

        weighted_task_losses = []
        for task_idx, ssl_task in enumerate(ssl_tasks):
            # 每个 SSL 任务都会返回一个长度为 N 的节点级损失向量。
            node_loss = ssl_task.compute_node_loss(adapted_model, data.x, data.edge_index)
            # 这里把该任务的节点损失乘上 gate 给它分配的节点级权重。
            weighted_task_losses.append(node_loss[ood_test_mask] * masked_gate_weights[:, task_idx])

        if weighted_task_losses:
            # 先在任务维度求和，再对节点求平均，得到最终反向传播的标量 loss。
            loss = torch.stack(weighted_task_losses, dim=0).sum(dim=0).mean()
        else:
            # 理论上不会走到这里，但保留兜底可以让脚本在空任务配置下也有明确行为。
            loss = torch.zeros((), device=data.x.device)

        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        # 每轮更新后都重新评估，并按 ood_test_acc 保存最优状态。
        current_metrics = evaluate_current(adapted_model)
        record_history(epoch, current_metrics)
        if not np.isnan(current_metrics["ood_test_acc"]) and current_metrics["ood_test_acc"] > best_ood_test:
            best_ood_test = float(current_metrics["ood_test_acc"])
            best_metrics = dict(current_metrics)
            best_model_state = copy.deepcopy(adapted_model.state_dict())
            best_gate_state = copy.deepcopy(gate.state_dict())

    # 训练结束后，把模型和 gate 恢复到训练期间最优的那一轮。
    adapted_model.load_state_dict(best_model_state)
    gate.load_state_dict(best_gate_state)
    adapted_model.eval()
    gate.eval()

    with torch.no_grad():
        # 重新计算最终 gate 权重统计，方便后续分析任务偏好。
        final_gate_weights = gate(pre_embed)

    if verbose:
        print("Shared encoder tasks: {}".format(", ".join("{}:{}".format(i + 1, name) for i, name in enumerate(task_names))))
        print(
            "[ttt_shared_onestage] gate init={} freeze_gate={} sum mean={:.6f}".format(
                str(gate_cfg["init"]),
                bool(gate_cfg["freeze_gate"]),
                float(final_gate_weights.sum(dim=-1).mean().item()),
            )
        )

    # artifacts 会被 main() 用来保存 checkpoint、指标曲线和参数文件。
    artifacts = {
        "model_cfg": model_cfg,
        "gate_cfg": gate_cfg,
        "gate_state_dict": gate.state_dict(),
        "classifier_state_dict": pre_model.classifier.state_dict(),
        "adapted_model_state_dict": adapted_model.state_dict(),
        "task_names": task_names,
        "task_cfg": task_cfg,
        "replace_last_k_layers": int(replace_last_k_layers),
        "history": history,
        "loss_weighting": "gate_weighted_sum_of_task_losses",
        "gate_weight_stats": {
            "task_weight_mean": final_gate_weights.mean(dim=0).detach().cpu().tolist(),
            "task_weight_min": final_gate_weights.min(dim=0).values.detach().cpu().tolist(),
            "task_weight_max": final_gate_weights.max(dim=0).values.detach().cpu().tolist(),
            "sum_mean": float(final_gate_weights.sum(dim=-1).mean().item()),
        },
    }
    return best_metrics if best_metrics is not None else initial_metrics, artifacts


def recompute_final_gate_weights(data, pre_ckpt, artifacts) -> torch.Tensor:
    model_cfg = artifacts["model_cfg"]
    gate_cfg = artifacts["gate_cfg"]
    pretrain_state_dict = pre_ckpt.get("model_state_dict")
    if pretrain_state_dict is None:
        pretrain_state_dict = pre_ckpt.get("state_dict")
    if pretrain_state_dict is None:
        raise KeyError("Pretrain checkpoint does not contain model_state_dict or state_dict")

    pre_model = GNNNodeClassifier(
        in_dim=int(model_cfg["in_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        gnn_type=str(model_cfg["gnn_type"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        use_bn=bool(model_cfg["use_bn"]),
        classifier_layers=int(model_cfg["classifier_layers"]),
    ).to(data.x.device)
    pre_model.load_state_dict(pretrain_state_dict)
    pre_model.eval()

    gate = GateMLP(
        in_dim=int(gate_cfg["in_dim"]),
        hidden_dim=int(gate_cfg["hidden_dim"]),
        num_layers=int(gate_cfg["num_layers"]),
        out_dim=int(gate_cfg["out_dim"]),
        dropout=float(gate_cfg["dropout"]),
        temperature=float(gate_cfg["temperature"]),
    ).to(data.x.device)
    gate.load_state_dict(artifacts["gate_state_dict"])
    gate.eval()

    with torch.no_grad():
        pre_embed = pre_model.get_embed(data.x, data.edge_index)
        return gate(pre_embed).detach().cpu()


def main():
    # 功能概括：脚本入口。
    # 顺序是：解析参数 -> 读取 pretrain -> 组装 stage2/stage3 风格配置 -> 训练/搜参 -> 保存结果。
    args = parse_args()
    if not args.pretrain_ckpt:
        raise ValueError("--pretrain-ckpt is required (or provide it via --params-file)")

    # 先固定随机种子，再决定运行设备，保证行为尽可能可复现。
    set_seed(int(args.seed))
    device = torch.device(args.device)

    # 读取 stage1 checkpoint，并补齐旧 checkpoint 里可能缺失的 model_cfg 字段。
    pre_ckpt_path = Path(args.pretrain_ckpt)
    pre_ckpt = safe_torch_load(pre_ckpt_path)
    pre_ckpt["model_cfg"] = normalize_pretrain_model_cfg(pre_ckpt, pre_ckpt_path)
    model_cfg = pre_ckpt["model_cfg"]

    # dataset/domain/shift 优先用命令行；没传时回退到 pretrain checkpoint 中保存的值。
    dataset = args.dataset or pre_ckpt.get("dataset", "")
    domain = args.domain or pre_ckpt.get("domain", "")
    shift = args.shift or pre_ckpt.get("shift", "")
    if not dataset or not domain or not shift:
        raise ValueError("dataset/domain/shift are required (in args or pretrain checkpoint)")

    # 这里开始是 stage2 风格的配置读取：任务名、任务配置、gate 结构。
    task_names = resolve_stage2_task_names(args.ssl_tasks, int(args.num_ssl))
    task_cfg = parse_task_cfg_json(args.task_cfg_json)

    # 数据加载沿用项目里统一的 GOOD 数据读取逻辑。
    data, masks, data_path = load_good_data(dataset, domain, shift, device)

    if args.use_optuna:
        if optuna is None:
            raise RuntimeError("Optuna is not installed, but --use-optuna is set")

        # 这几个变量用来记录 Optuna 搜索过程中表现最好的 trial。
        best_trial_score = float("-inf")
        best_trial_metrics = None
        best_trial_artifacts = None
        best_train_cfg = None

        def objective(trial):
            # 功能概括：Optuna 的单次试验。
            # 这里的搜索空间完全沿用原版 stage3 shared：
            # ssl_lr / encoder_lr / weight_decay / replace_last_k_layers；
            # gate_temperature 也在这里按 trial 单独构建 gate_cfg。
            nonlocal best_trial_score, best_trial_metrics, best_trial_artifacts, best_train_cfg
            train_cfg = build_trial_train_cfg(trial, args, model_cfg)
            gate_cfg_tmp = build_trial_gate_cfg(trial, args, model_cfg, len(task_names))
            metrics_tmp, artifacts_tmp = run_ttt_shared_onestage_once(
                args=args,
                data=data,
                masks=masks,
                pre_ckpt=pre_ckpt,
                task_names=task_names,
                task_cfg=task_cfg,
                gate_cfg=gate_cfg_tmp,
                train_cfg=train_cfg,
                verbose=False,
            )
            # 搜索目标仍然是最大化 ood_test_acc，这和原版 stage3 一致。
            score = metrics_tmp["ood_test_acc"]
            if np.isnan(score):
                score = -1.0
            if score > best_trial_score:
                best_trial_score = float(score)
                best_trial_metrics = copy.deepcopy(metrics_tmp)
                best_trial_artifacts = copy.deepcopy(artifacts_tmp)
                best_train_cfg = dict(train_cfg)
            return score

        # timeout <= 0 时不设时间上限；否则按秒限制总搜索时间。
        timeout = None if int(args.optuna_timeout) <= 0 else int(args.optuna_timeout)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=int(args.optuna_trials), timeout=timeout)
        if best_trial_metrics is None or best_trial_artifacts is None or best_train_cfg is None:
            raise RuntimeError("Optuna finished without a valid best trial result to save.")
        metrics = best_trial_metrics
        artifacts = best_trial_artifacts
    else:
        # 不开 Optuna 时，直接用命令行里传入的 stage3 风格训练参数跑一次。
        gate_cfg = build_gate_cfg(args, model_cfg, len(task_names))
        best_train_cfg = build_base_train_cfg(args, model_cfg)
        metrics, artifacts = run_ttt_shared_onestage_once(
            args=args,
            data=data,
            masks=masks,
            pre_ckpt=pre_ckpt,
            task_names=task_names,
            task_cfg=task_cfg,
            gate_cfg=gate_cfg,
            train_cfg=best_train_cfg,
            verbose=True,
        )

    gate_cfg = artifacts["gate_cfg"]

    # 下面开始是统一的结果落盘逻辑：创建目录、存模型、存指标、存参数、存曲线。
    out_dir = make_stage_output_dir(
        args.output_root,
        "stage23_shared_onestage",
        dataset,
        domain,
        shift,
        timestamp=args.timestamp,
    )
    ckpt_path = out_dir / "ttt_shared_onestage_model.pt"
    gate_ckpt_path = out_dir / "gate_model.pt"
    gate_weights_path = out_dir / "gate_node_weights.pt"
    metrics_path = out_dir / "metrics.json"
    params_path = out_dir / "params.json"
    plot_path = out_dir / "ood_test_acc_vs_epoch.png"
    gate_plot_path = out_dir / "plot.png"

    final_gate_weights_cpu = recompute_final_gate_weights(data, pre_ckpt, artifacts)

    # history 在训练过程中已经记录好了，这里直接画 OOD test acc 曲线。
    save_ttt_curve_plot(artifacts.get("history", []), plot_path)
    save_gate_weight_plot(artifacts["task_names"], final_gate_weights_cpu, gate_plot_path)

    # checkpoint 保存的是复现实验所需的完整状态：模型、gate、配置、指标、历史曲线数据。
    torch.save(
        {
            "stage": "ttt_shared_onestage",
            "pretrain_ckpt": str(args.pretrain_ckpt),
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
            "adapted_model_state_dict": artifacts["adapted_model_state_dict"],
            "loss_weighting": artifacts.get("loss_weighting", ""),
            "gate_weight_stats": artifacts.get("gate_weight_stats", {}),
            "train_cfg": best_train_cfg,
            "metrics": metrics,
            "history": artifacts.get("history", []),
        },
        ckpt_path,
    )

    torch.save(
        {
            "stage": "ttt_shared_onestage_gate",
            "pretrain_ckpt": str(args.pretrain_ckpt),
            "dataset": dataset,
            "domain": domain,
            "shift": shift,
            "data_path": str(data_path),
            "model_cfg": artifacts["model_cfg"],
            "task_names": artifacts["task_names"],
            "task_cfg": artifacts["task_cfg"],
            "gate_cfg": artifacts["gate_cfg"],
            "gate_state_dict": artifacts["gate_state_dict"],
            "gate_weight_stats": artifacts.get("gate_weight_stats", {}),
            "metrics": metrics,
        },
        gate_ckpt_path,
    )

    torch.save(
        {
            "stage": "ttt_shared_onestage_gate_weights",
            "dataset": dataset,
            "domain": domain,
            "shift": shift,
            "task_names": artifacts["task_names"],
            "weights": final_gate_weights_cpu,
            "ood_test_mask": masks["ood_test"].detach().cpu(),
        },
        gate_weights_path,
    )

    # metrics.json 只保留最终指标，方便批处理脚本快速读取。
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # params.json 则保存这次运行的主要超参数，便于你后面回看和复现。
    run_params = {
        "params_file": args.params_file,
        "dataset": dataset,
        "domain": domain,
        "shift": shift,
        "pretrain_ckpt": str(args.pretrain_ckpt),
        "ssl_tasks": ",".join(task_names),
        "num_ssl": int(len(task_names)),
        "task_cfg_json": json.dumps(task_cfg, ensure_ascii=True),
        "ssl_lr": float(best_train_cfg["ssl_lr"]),
        "encoder_lr": float(best_train_cfg["encoder_lr"]),
        "gate_lr": float(best_train_cfg["gate_lr"]),
        "weight_decay": float(best_train_cfg["weight_decay"]),
        "finetune_epochs": int(best_train_cfg["finetune_epochs"]),
        "replace_last_k_layers": int(best_train_cfg["replace_last_k_layers"]),
        "gate_hidden_dim": int(gate_cfg["hidden_dim"]),
        "gate_num_layers": int(gate_cfg["num_layers"]),
        "gate_dropout": float(gate_cfg["dropout"]),
        "gate_temperature": float(gate_cfg["temperature"]),
        "gate_temperature_min": float(args.gate_temperature_min),
        "gate_temperature_max": float(args.gate_temperature_max),
        "gate_init": str(gate_cfg["init"]),
        "gate_init_scale": float(gate_cfg["init_scale"]),
        "freeze_gate": bool(gate_cfg["freeze_gate"]),
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
        "gate_model_path": str(gate_ckpt_path),
        "gate_plot_path": str(gate_plot_path),
        "gate_node_weights_path": str(gate_weights_path),
    }
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)

    print("Saved ttt_shared_onestage checkpoint:", ckpt_path)
    print("Saved gate checkpoint:", gate_ckpt_path)
    print("Saved gate node weights:", gate_weights_path)
    print("Saved params:", params_path)
    print("Saved plot:", plot_path)
    print("Saved gate plot:", gate_plot_path)
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    # 作为脚本运行时，从这里进入主流程。
    main()
