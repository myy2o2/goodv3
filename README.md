# GOOD OOD TTT Workflow

这个仓库当前主流程只有两个阶段：

1. `pretrain.py`：训练 Stage 1 的监督 GNN 节点分类器。
2. `tttnew.py`：从 Stage 1 checkpoint 出发，执行带 gate 的 test-time training。

`templates/stage3_*.json` 现在作为 `tttnew.py` 的参数文件使用。

## 数据目录

脚本默认从 `./datasets` 读取 GOOD 数据，路径格式如下：

```text
datasets/<GOODDataset>/<domain>/processed/<shift>.pt
```

当前脚本支持的常用配置：

| Dataset | Domain | Shift | Stage 1 params | Stage 2 params |
| --- | --- | --- | --- | --- |
| Citeseer | degree | concept | `templates/stage1_cit_con_deg.json` | `templates/stage2_cit_con_deg.json` |
| Citeseer | word | covariate | `templates/stage1_cit_cov_word.json` | `templates/stage2_cit_cov_word.json` |
| Cora | degree | concept | `templates/stage1_cora_con_deg.json` | `templates/stage2_cora_con_deg.json` |
| Cora | word | covariate | `templates/stage1_cora_cov_word.json` | `templates/stage2_cora_cov_word.json` |
| Pubmed | degree | concept | `templates/stage1_pub_con_deg.json` | `templates/stage2_pub_con_deg.json` |
| Pubmed | word | covariate | `templates/stage1_pub_cov_word.json` | `templates/stage2_pub_cov_word.json` |
| WikiCS | degree | concept | `templates/stage1_wikics_con_deg.json` | `templates/stage2_wikics_con_deg.json` |
| WikiCS | word | covariate | `templates/stage1_wikics_cov_word.json` | `templates/stage2_wikics_cov_word.json` |
| Arxiv | degree | concept | `templates/stage1_arxiv_con_deg.json` | `templates/stage2_arxiv_con_deg.json` |
| Arxiv | time | covariate | `templates/stage1_arxiv_cov_time.json` | `templates/stage2_arxiv_cov_time.json` |

## Stage 1: Pretrain

Stage 1 训练输出目录为：

```text
outputs/stage1/<dataset>/<domain>/<shift>/<timestamp>/
```

主要输出文件：

```text
pretrain_model.pt
metrics.json
params.json
```

推荐把 `timestamp` 固定为 `right`，这样 Stage 2 参数文件里的默认 `pretrain_ckpt` 路径可以直接命中：

```bash
python pretrain.py --params-file templates/stage1_cit_con_deg.json --timestamp right
python pretrain.py --params-file templates/stage1_cit_cov_word.json --timestamp right
python pretrain.py --params-file templates/stage1_cora_con_deg.json --timestamp right
python pretrain.py --params-file templates/stage1_cora_cov_word.json --timestamp right
python pretrain.py --params-file templates/stage1_pub_con_deg.json --timestamp right
python pretrain.py --params-file templates/stage1_pub_cov_word.json --timestamp right
python pretrain.py --params-file templates/stage1_wikics_con_deg.json --timestamp right
python pretrain.py --params-file templates/stage1_wikics_cov_word.json --timestamp right
python pretrain.py --params-file templates/stage1_arxiv_con_deg.json --timestamp right
python pretrain.py --params-file templates/stage1_arxiv_cov_time.json --timestamp right
```


```bash
python tttnew.py \
  --params-file templates/stage2_cit_con_deg.json \
  --pretrain-ckpt outputs/stage1/citeseer/degree/concept/<timestamp>/pretrain_model.pt
```

## Stage 2: TTTNew

Stage 2 使用 `tttnew.py`，输入是 Stage 1 的 `pretrain_model.pt`。它会读取参数文件里的：

- `pretrain_ckpt`
- `dataset` / `domain` / `shift`
- `ssl_tasks` / `num_ssl` / `task_cfg_json`
- gate 配置，例如 `gate_hidden_dim`、`gate_num_layers`、`gate_dropout`、`gate_temperature`
- TTT 训练配置，例如 `ssl_lr`、`encoder_lr`、`gate_lr`、`weight_decay`、`finetune_epochs`

Stage 2 输出目录为：

```text
outputs/stage23_shared_onestage/<dataset>/<domain>/<shift>/<timestamp>/
```

主要输出文件：

```text
ttt_shared_onestage_model.pt
gate_model.pt
gate_node_weights.pt
metrics.json
params.json
ood_test_acc_vs_epoch.png
```

运行全部配置：

```bash
python tttnew.py --params-file templates/stage2_cit_con_deg.json
python tttnew.py --params-file templates/stage2_cit_cov_word.json
python tttnew.py --params-file templates/stage2_cora_con_deg.json
python tttnew.py --params-file templates/stage2_cora_cov_word.json
python tttnew.py --params-file templates/stage2_pub_con_deg.json
python tttnew.py --params-file templates/stage2_pub_cov_word.json
python tttnew.py --params-file templates/stage2_wikics_con_deg.json
python tttnew.py --params-file templates/stage2_wikics_cov_word.json
python tttnew.py --params-file templates/stage2_arxiv_con_deg.json
python tttnew.py --params-file templates/stage2_arxiv_cov_time.json
```

单个配置的完整例子：

```bash
python pretrain.py --params-file templates/stage1_cit_con_deg.json --timestamp right
python tttnew.py --params-file templates/stage2_cit_con_deg.json
```

## 参数文件约定

Stage 1 参数文件命名：

```text
templates/stage1_<dataset>_<shift-abbrev>_<domain-abbrev>.json
```

Stage 2 参数文件命名：

```text
templates/stage2_<dataset>_<shift-abbrev>_<domain-abbrev>.json
```

这里的 `stage2` 指当前第二阶段 `tttnew.py`，不是旧版单独 gate 训练。

常用缩写：

| Abbrev | Meaning |
| --- | --- |
| `cit` | `citeseer` |
| `pub` | `pubmed` |
| `con` | `concept` |
| `cov` | `covariate` |
| `deg` | `degree` |


