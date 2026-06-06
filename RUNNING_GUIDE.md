# Running Guide

完整两阶段流程已经整理到 `README.md`。

当前主流程：

```bash
python pretrain.py --params-file templates/stage1_pub_cov_word.json --timestamp right
python tttnew.py --params-file templates/stage2_pub_cov_word.json
```

其它数据集和 shift 的命令见 `README.md` 中的 Stage 1 和 Stage 2 命令列表。
