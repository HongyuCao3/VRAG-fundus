import argparse
import torch
import os
import json
from tqdm import tqdm

class DataExtractor():
    def __init__(self):
        pass
    
    def extract_image_data_crop(self, json_folder):
        image_data = []

        # 遍历指定目录下的所有JSON文件
        for filename in os.listdir(json_folder):
            if filename.endswith('.json'):
                file_path = os.path.join(json_folder, filename)
                with open(file_path, 'r') as f:
                    # 加载JSON内容
                    data = json.load(f)
                    
                    # 提取信息并构成元组
                    for text, image_path in data.items():
                        meta_data = {'source_file': filename}  # 可以根据需要添加更多元数据
                        image_data.append((image_path, text, meta_data))

        return image_data
    
    def extract_image_data_level(self, json_folder):
        image_data = []

        # 遍历指定目录下的所有JSON文件
        for filename in os.listdir(json_folder):
            if filename.endswith('.json'):
                file_path = os.path.join(json_folder, filename)
                with open(file_path, 'r') as f:
                    # 加载JSON内容
                    data = json.load(f)
                    
                    # 提取信息并构成元组
                    for item in data:
                        text = item["dis"]
                        image_path = item["image_path"]
                        meta_data = item["imid"]  # 可以根据需要添加更多元数据
                        image_data.append((image_path, text, meta_data))

        return image_data
    
    def extract_image_data_classic(self, folder):
        image_data = []
        # 支持的图片格式列表
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        
        # 获取根目录的绝对路径
        base_dir = os.path.abspath(folder)
        
        for root, dirs, files in os.walk(folder):
            for file in files:
                # 获取文件扩展名
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    # 构建完整的文件路径
                    full_path = os.path.join(root, file)
                    
                    # 获取子文件夹名称
                    sub_folder = os.path.relpath(root, base_dir)
                    
                    # 如果子文件夹是根目录，则显示为空字符串
                    if sub_folder == '.':
                        sub_folder = ''
                    if sub_folder == "DR":
                        text = file.split("_")[0]
                        if text == "no DR":
                            text = "Normal"
                    elif sub_folder == "metaPM":
                        text = file.split("_")[0]
                    else:
                        text = sub_folder
                    image_path = full_path
                    meta_data = file
                    image_data.append((image_path, text, meta_data))
                    print(f"Image found: {file} in subfolder: {sub_folder} at path: {full_path}")
        return image_data