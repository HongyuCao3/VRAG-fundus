import torch
import os, json, sys
from transformers import CLIPModel, CLIPProcessor
from abc import ABC
from PIL import Image
from data_extractor import DataExtractor
import argparse
from torch.nn.functional import cosine_similarity
sys.path.append(r"/home/hongyu/Visual-RAG-LLaVA-Med/")

class BaseEmbBuilder(ABC):
    def __init__(self):
        pass
    
    def find_json_file(self, folder):
        """
        查找指定文件夹中的JSON文件。
        
        :param folder: 要查找的文件夹路径
        :return: JSON文件的路径，如果未找到则返回None
        """
        for filename in os.listdir(folder):
            if filename.endswith('.json'):
                return os.path.join(folder, filename)
        return None
    
    def get_layer_representation(self, img_path: str, layer_index: int=11):
        # 加载并预处理图像
        image = Image.open(img_path)
        inputs = self.processor(images=image, return_tensors="pt")

        # 获取模型的视觉编码器
        visual_model = self.model.vision_model

        # 定义一个前向传播钩子来捕获指定层的输出
        layer_output = None

        def hook(module, input, output):
            nonlocal layer_output
            if isinstance(output, tuple):
                layer_output = output[0]  # 如果输出是元组，取第一个元素
            else:
                layer_output = output

        # 获取目标层
        target_layer = visual_model.encoder.layers[layer_index]

        # 在指定层注册钩子
        handle = target_layer.register_forward_hook(hook)

        # 前向传播
        with torch.no_grad():
            _ = visual_model(pixel_values=inputs['pixel_values'])

        # 移除钩子
        handle.remove()

        return layer_output
    
    def calculate_similarity(self, img_path1: str, img_path2: str, layer_index: int=11):
        """get similarity of two images
        """
        # 获取两张图片的特征表示
        feature1 = self.get_layer_representation(img_path1, layer_index)
        feature2 = self.get_layer_representation(img_path2, layer_index)

        # 检查特征张量的形状
        print(f"Feature1 shape: {feature1.shape}")
        print(f"Feature2 shape: {feature2.shape}")

        # 根据特征张量的形状进行平均池化
        if len(feature1.shape) == 4:
            feature1 = feature1.mean(dim=(2, 3))
            feature2 = feature2.mean(dim=(2, 3))
        elif len(feature1.shape) == 3:
            feature1 = feature1.mean(dim=2)
            feature2 = feature2.mean(dim=2)
        else:
            raise ValueError("Unexpected feature tensor shape")

        # 计算余弦相似度
        similarity = cosine_similarity(feature1, feature2, dim=1)

        return similarity.item()  # 返回相似度的标量值
    
    def save_image_representations(self, source_folder: str, target_folder: str, layer_index: int=11):
        # TODO:修改输入为image_data: list
        """
        """
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        representation_data = {}
        for filename in os.listdir(source_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(source_folder, filename)
                representation = self.get_layer_representation(file_path, layer_index)
                
                # Save the representation as a .pt file
                representation_file = os.path.join(target_folder, f"{os.path.splitext(filename)[0]}.pt")
                torch.save(representation, representation_file)

                # Record the correspondence
                representation_data[filename] = representation_file

        # Save the correspondence data to a JSON file
        correspondence_file = os.path.join(target_folder, 'correspondence.json')
        with open(correspondence_file, 'w') as f:
            json.dump(representation_data, f)