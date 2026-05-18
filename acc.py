import os
import re

stage2_dir = 'outputs/stage2/pubmed/degree/concept'
stage3_dir = 'outputs/stage3/pubmed/degree/concept'

# 1. 提取所有单task的acc
task_acc = {}
for name in os.listdir(stage3_dir):
    if os.path.isdir(os.path.join(stage3_dir, name)):
        m = re.match(r'(\d{4})_(\w+)$', name)
        if m:
            acc, task = m.groups()
            acc_val = float(acc[:2] + '.' + acc[2:])  # 9912 -> 99.12
            task_acc[task] = acc_val

# 2. 遍历stage2，处理多task组合
mapping = []
for name in os.listdir(stage2_dir):
    old_path = os.path.join(stage2_dir, name)
    if os.path.isdir(old_path):
        m = re.match(r'(.+)_\d+', name)
        if m:
            task_combo = m.group(1)
            tasks = task_combo.split('-')
            accs = []
            for t in tasks:
                if t in task_acc:
                    accs.append(task_acc[t])
            if len(accs) == len(tasks):
                mean_acc = sum(accs) / len(accs)
                acc_val = 100 - mean_acc
                acc_str = f"{acc_val:.2f}".replace('.', '').zfill(4)
                new_name = f"{acc_str}_{task_combo}"
                new_path = os.path.join(stage2_dir, new_name)
                mapping.append((old_path, new_path))

# 3. 重命名并输出对应关系
for old, new in mapping:
    if not os.path.exists(new):
        os.rename(old, new)
        print(f"{os.path.basename(old)}  -->  {os.path.basename(new)}")
    else:
        print(f"目标已存在，跳过: {os.path.basename(new)}")