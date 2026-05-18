import os
import json
import sys

"""
用法：
    python rename_by_ood_test_acc.py <目标父目录>

功能：
    遍历目标父目录下的所有子文件夹，
    对每个子文件夹，读取其 metrics.json 文件，
    取出 ood_test_acc，计算 1-ood_test_acc，
    然后将该子文件夹重命名为小数点后四位（去掉前导 0.），如 0.1234 -> 1234。
    如果新名字已存在，则跳过。
"""

def main(root_dir):
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if not folder.startswith("batch"):
            # 只处理以batch开头的文件夹
            continue
        metrics_path = os.path.join(folder_path, "metrics.json")
        if not os.path.isfile(metrics_path):
            print(f"跳过 {folder_path}，未找到 metrics.json")
            continue
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            ood_acc = metrics["ood_test_acc"]
            new_val = 1 - float(ood_acc)
            # 保留小数点后4位，去掉前导 0.，并加上原始文件夹名
            prefix = f"{new_val:.4f}"[2:]
            new_name = f"{prefix}_{folder}"
            new_folder_path = os.path.join(root_dir, new_name)
            if os.path.exists(new_folder_path):
                print(f"目标文件夹 {new_name} 已存在，跳过 {folder}")
                continue
            os.rename(folder_path, new_folder_path)
            print(f"{folder} -> {new_name}")
        except Exception as e:
            print(f"处理 {folder_path} 时出错: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python rename_by_ood_test_acc.py <目标父目录>")
        sys.exit(1)
    main(sys.argv[1])
