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
from llama_index.embeddings.openai import OpenAIEmbedding

class IndexBuilder():
    def __init__(self, args):
        self.crop_dir = args.crop_dir
        self.level_dir = args.level_dir
        self.classic_dir = args.classic_dir
        self.persist_dir = args.persist_dir
        self.embedding_name = args.embedding_name
        # Settings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_name)
        
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
            document = self.process_images(document)
            image_nodes_ = [ImageNode(image_path=p, text=t, meta_data=k) for p, t, k in document]
            image_nodes.extend(image_nodes_)
            
        # read classic data
        if self.classic_dir != None:
            document = self.extract_image_data_classic_dr(self.classic_dir)
            image_nodes_ = [ImageNode(image_path=p, text=t, meta_data=k) for p, t, k in document]
            image_nodes.extend(image_nodes_)
        
        # use llama-index to construct index
        print(self.embedding_name)
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_name)
        self.multi_index = MultiModalVectorStoreIndex(image_nodes, show_progress=True, embed_model=HuggingFaceEmbedding(model_name=self.embedding_name))
        
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
    
    def process_images(self, image_data):
        """
        对给定的image_data列表中的每个图像执行旋转(90°, 180°, 270°)和水平翻转，
        并将新的图像保存到原始路径的同一目录下，同时更新image_data列表。
        
        :param image_data: 列表，其中每个元素是一个包含(image_path, text, meta_data)的三元组
        :return: 更新后的image_data列表
        """
        transformations = [
            (Image.ROTATE_90, '_rotate_90'),
            (Image.ROTATE_180, '_rotate_180'),
            (Image.ROTATE_270, '_rotate_270'),
            (Image.FLIP_LEFT_RIGHT, '_flip_horizontal')
        ]
        image_data_ = []
        for image_path, text, meta_data in tqdm(image_data):
            # 打开原始图像
            with Image.open(image_path) as img:
                # 获取图像所在的目录和文件名
                directory, filename = os.path.split(image_path)
                name, ext = os.path.splitext(filename)
                
                # 应用每种变换
                for transform, suffix in transformations:
                    transformed_img = img.transpose(transform)
                    new_filename = f"{name}{suffix}{ext}"
                    new_image_path = os.path.join(directory, new_filename)
                    
                    # 保存新图像
                    transformed_img.save(new_image_path)
                    
                    # 将新图像的信息添加到image_data中
                    image_data_.append((new_image_path, text, meta_data))
        image_data.extend(image_data_)
        return image_data
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-dir", type=str, default=None)
    parser.add_argument("--level-dir", type=str, default=None)
    parser.add_argument("--embedding-name", type=str, default=None)
    parser.add_argument("--classic-dir", type=str, default=None)
    parser.add_argument("--persist-dir", type=str, default=None)
    args = parser.parse_args()
    IB = IndexBuilder(args)
    IB.build_index()