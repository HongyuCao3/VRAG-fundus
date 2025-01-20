import os
import csv
import shutil
from collections import defaultdict
import random

root_dir = "/home/hongyu"
# 源数据路径
source_dir = root_dir + '/alldataset/'
source_csv = os.path.join(source_dir, 'cleaned_full.csv')
source_images_dir = os.path.join(source_dir, 'images')

# 目标数据路径
target_dir = root_dir + '/partdataset/'
os.makedirs(target_dir, exist_ok=True)
target_csv = os.path.join(target_dir, 'cleaned_part.csv')
target_images_dir = os.path.join(target_dir, 'images')
os.makedirs(target_images_dir, exist_ok=True)

# 读取源CSV文件并按类别分类
category_samples = defaultdict(list)
with open(source_csv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        category = row['folder_standardized']
        category_samples[category].append(row)

# 随机选取每类两个样本
selected_samples = []
for category, samples in category_samples.items():
    # 如果该类别的样本数少于2个，则全部选取
    selected = random.sample(samples, min(2, len(samples)))
    selected_samples.extend(selected)

# 将选中的样本信息写入新的CSV文件
fieldnames = ['imid', 'folder_standardized', 'Finding', 'modality', 'dataset', 'caption']
with open(target_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for sample in selected_samples:
        writer.writerow(sample)
        # 复制图像文件
        source_image_path = os.path.join(source_images_dir, f"{sample['imid']}.jpg")
        target_image_path = os.path.join(target_images_dir, f"{sample['imid']}.jpg")
        shutil.copy2(source_image_path, target_image_path)

print(f"Selected {len(selected_samples)} samples and copied them to {target_dir}")