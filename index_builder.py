import argparse
import torch
import os
import json
from tqdm import tqdm

from PIL import Image
from transformers import set_seed, logging

from utils import split_image, delete_images

from llama_index.core import (ServiceContext, 
                               SimpleDirectoryReader,
                               SimpleDirectoryReader,
                               StorageContext,
                               load_index_from_storage,
                               Settings)
from llama_index.core.schema import ImageNode
from llama_index.core.schema import ImageDocument
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class IndexBuilder():
    def __init__(self, args):
        self.crop_dir = args.crop_dir
        self.level_dir = args.level_dir
        self.classic_dir = args.classic_dir
        self.persist_dir = args.persist_dir
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
    def build_index(self,):
        document = []
        image_nodes = []
        
        # read crop data
        if self.crop_dir != None:
            document = self.extract_image_data_crop(self.crop_dir)
            image_nodes = [ImageNode(image_path=self.crop_dir+p, text=t, meta_data=k) for p, t, k in document]
        
        # read level data
        if self.level_dir != None:
            document = self.extract_image_data_level(self.level_dir)
            image_nodes_ = [ImageNode(image_path=p, text=t, meta_data=k) for p, t, k in document]
            image_nodes.extend(image_nodes_)
            
        # read classic data
        if self.classic_dir != None:
            document = self.extract_image_data_classic(self.classic_dir)
            image_nodes_ = [ImageNode(image_path=p, text=t, meta_data=k) for p, t, k in document]
            image_nodes.extend(image_nodes_)
        
        # use llama-index to construct index
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.multi_index = MultiModalVectorStoreIndex(image_nodes, show_progress=True)
        
        # save index
        if self.persist_dir != None:
            self.multi_index.storage_context.persist(persist_dir=self.persist_dir)
        else:
            print("Error save path")
        
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
    
    def extract_image_data_classic_dr(self, folder):
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
                        elif text == "mild NPDR":
                            text = "mild nonproliferative diabetic retinopathy"
                        elif text == "PDR":
                            text = "proliferative diabetic retinopathy"
                        elif text == "severe NPDR":
                            text = "severe nonproliferative diabetic retinopathy"
                        elif text == "moderate" or text == "moderate NPDR":
                            text = "moderate nonproliferative diabetic retinopathy"
                        image_path = full_path
                        meta_data = file
                        image_data.append((image_path, text, meta_data))
                        print(f"Image found: {file} in subfolder: {sub_folder} at path: {full_path}")
        return image_data
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-dir", type=str, default=None)
    parser.add_argument("--level-dir", type=str, default=None)
    parser.add_argument("--classic-dir", type=str, default=None)
    parser.add_argument("--persist-dir", type=str, default=None)
    args = parser.parse_args()
    IB = IndexBuilder(args)
    IB.build_index()