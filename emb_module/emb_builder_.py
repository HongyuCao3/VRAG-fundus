import torch
import os, json
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from data_extractor import DataExtractor
import argparse
from torch.nn.functional import cosine_similarity
from utils import find_json_file, convert_abbreviation_to_full_name

class EmbBuilder():
    def __init__(self, img_folder, emb_folder):
        self.img_path = img_folder
        self.emb_folder = emb_folder
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.json_file = find_json_file(self.img_path)
        self.data_extractor = DataExtractor()
        if self.json_file is None:
            raise FileNotFoundError("JSON file not found in the specified folder.")
        
    def get_layer_representation(self, img_path, layer_index=11):
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

    def calculate_similarity(self, img_path1, img_path2, layer_index=11):
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
    
    def load_image_representations(self, target_folder):
        correspondence_file = os.path.join(target_folder, 'correspondence.json')
        with open(correspondence_file, 'r') as f:
            representation_data = json.load(f)
        
        representations = {}
        for img_name, rep_path in representation_data.items():
            representation = torch.load(rep_path)
            representations[img_name] = representation
        
        return representations
    
class CropEmbBuilder(EmbBuilder):
    def __init__(self, img_folder, emb_folder):
        super().__init__(img_folder, emb_folder)
        
    def get_detailed_similarities(self, input_img, k=5):
        """
        获取输入图片与文件夹中保存的嵌入的相似度，并返回最相似的前k个图像的详细信息。
        
        :param input_img: 输入图片的路径
        :param emb_folder: 包含预计算嵌入和对应关系的文件夹路径
        :param json_file: 包含图像详细信息的JSON文件路径
        :param k: 返回最相似图像的数量，默认为5
        :return: 一个列表，包含最相似图像的score, dis, 和 imid
        """
        # 获取最相似的图像
        similar_images = self.find_similar_images(input_img, k=k)


        # 获取详细的相似信息
        score_ = []
        txt_ = []
        metadata_ = []
        img_ = []
        for img_path, score in similar_images:
            score_.append(score)
            txt_.append(img_path.split("/")[-1].split(".")[0])
            metadata_.append(img_path)
            img_.append("." + ".".join(img_path.split(".")[-2:]))
        detailed_similarities = {"score":score_, "txt": txt_, "metadata": metadata_, "img": img_}

        return detailed_similarities
    
    def get_detailed_similarities_str(self, input_img, k=5, lesion_str=None):
        """
        获取输入图片与文件夹中保存的嵌入的相似度，并返回最相似的前k个图像的详细信息。

        :param input_img: 输入图片的路径
        :param k: 返回最相似图像的数量，默认为5
        :param lesion_str: 只考虑文件名中包含此字符串的图像
        :return: 一个列表，包含最相似图像的score, dis, 和 imid
        """
        # 获取最相似的图像
        similar_images = self.find_similar_images_str_crop(input_img, k=k, lesion_str=lesion_str)

        # 获取详细的相似信息
        score_ = []
        txt_ = []
        metadata_ = []
        img_ = []
        for img_path, score in similar_images:
            score_.append(score)
            txt_.append(img_path.split("/")[-1].split(".")[0])
            metadata_.append(img_path)
            img_.append("." + ".".join(img_path.split(".")[-2:]))
        detailed_similarities = {"score": score_, "txt": txt_, "metadata": metadata_, "img": img_}

        return detailed_similarities
    
    def process_lesion_data(self, source_root, target_folder, layer_index=11):
        """
        处理病变数据集，提取每张图片的特征表示并保存。

        参数:
            source_root (str): 源数据根目录路径。
            target_folder (str): 目标保存目录路径。
            layer_index (int): 特征提取使用的模型层索引。
        """
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        representation_data = {}

        # 获取所有JSON文件
        json_files = [f for f in os.listdir(source_root) if f.endswith('.json')]

        for json_file in json_files:
            json_path = os.path.join(source_root, json_file)
            with open(json_path, 'r') as f:
                images_dict = json.load(f)

            # 对JSON文件中指定的每张图片进行处理
            for key, value in images_dict.items():
                # 构建正确的图片路径
                image_folder = key  # 使用键作为文件夹名
                image_path = os.path.join(source_root, value)
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} does not exist.")
                    continue

                representation = self.get_layer_representation(image_path, layer_index)

                # 确保文件名唯一，可以采用JSON文件名+原文件名的形式
                unique_name = f"{os.path.splitext(json_file)[0]}_{os.path.splitext(key)[0]}"
                representation_file = os.path.join(target_folder, f"{unique_name}.pt")
                torch.save(representation, representation_file)

                # 记录对应关系
                representation_data[image_path] = representation_file

        # 保存所有图片的对应关系到JSON文件
        correspondence_file = os.path.join(target_folder, 'correspondence.json')
        with open(correspondence_file, 'w') as f:
            json.dump(representation_data, f)
            
    
    def filter_images_by_lesion(self, lesion_str):
        """
        根据指定字符串过滤图片。
        
        :param lesion_str: 需要在详细信息中包含的字符串
        :return: 过滤后的图片名称列表
        """
        with open(self.json_file, 'r') as f:
            image_details = json.load(f)
        
        filtered_images = [
            detail['image_path']
            for detail in image_details
            if lesion_str.lower() in detail["dis"].lower()
        ]
        return filtered_images
    
    def find_similar_images_str(self, input_img, k=2, layer=11, lesion_str=None):
        """
        对于输入图片，计算其与emb_folder中保存的所有嵌入的相似度，
        并返回最相似的前k个图像的原始路径。

        :param input_img: 输入图片的路径
        :param k: 返回最相似图像的数量，默认为5
        :param layer: 使用的神经网络层，默认为11
        :param lesion_str: 只考虑文件名中包含此字符串的图像
        :return: 一个列表，包含最相似图像的原始路径和相似度分数
        """
        # 获取输入图片的嵌入
        input_emb = self.get_layer_representation(input_img, layer_index=layer)
        
        # 确保输入嵌入是一个二维张量 (batch_size, feature_dim)
        if len(input_emb.shape) == 4:
            input_emb = input_emb.mean(dim=(2, 3))  # 全局平均池化
        elif len(input_emb.shape) == 3:
            input_emb = input_emb.mean(dim=2)  # 全局平均池化
        else:
            raise ValueError("Unexpected feature tensor shape")

        # 加载文件夹中的所有嵌入，并根据lesion_str过滤
        representations = self.load_image_representations(self.emb_folder)
        if lesion_str is not None:
            representations = {img_name: rep for img_name, rep in representations.items() if lesion_str in img_name}

        # 计算所有嵌入与输入嵌入的相似度
        similarities = []
        for img_name, rep in representations.items():
            # 确保预加载的嵌入也是一个二维张量 (batch_size, feature_dim)
            if len(rep.shape) == 4:
                rep = rep.mean(dim=(2, 3))  # 全局平均池化
            elif len(rep.shape) == 3:
                rep = rep.mean(dim=2)  # 全局平均池化
            else:
                raise ValueError("Unexpected feature tensor shape")

            # 计算余弦相似度
            sim = cosine_similarity(input_emb, rep, dim=1)
            similarities.append((img_name, sim.item()))

        # 按相似度排序并选择前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        # 获取最相似图像的原始路径
        similar_images = [(os.path.join(self.img_path, img_name), sim) for img_name, sim in top_k]
        return similar_images
    
    
class LevelEmbBuilder(EmbBuilder):
    def __init__(self, img_folder, emb_folder):
        super().__init__(img_folder, emb_folder)
    
    def find_similar_images(self, input_img, k=2, layer=11):
        """
        对于输入图片，计算其与emb_folder中保存的所有嵌入的相似度，
        并返回最相似的前k个图像的原始路径。
        
        :param input_img: 输入图片的路径
        :param emb_folder: 包含预计算嵌入和对应关系的文件夹路径
        :param k: 返回最相似图像的数量，默认为5
        :return: 一个列表，包含最相似图像的原始路径和相似度分数
        """
        # 获取输入图片的嵌入
        input_emb = self.get_layer_representation(input_img, layer_index=layer)
        
        # 确保输入嵌入是一个二维张量 (batch_size, feature_dim)
        if len(input_emb.shape) == 4:
            input_emb = input_emb.mean(dim=(2, 3))  # 全局平均池化
        elif len(input_emb.shape) == 3:
            input_emb = input_emb.mean(dim=2)  # 全局平均池化
        else:
            raise ValueError("Unexpected feature tensor shape")

        # 加载文件夹中的所有嵌入
        representations = self.load_image_representations(self.emb_folder)

        # 计算所有嵌入与输入嵌入的相似度
        similarities = []
        for img_name, rep in representations.items():
            # 确保预加载的嵌入也是一个二维张量 (batch_size, feature_dim)
            if len(rep.shape) == 4:
                rep = rep.mean(dim=(2, 3))  # 全局平均池化
            elif len(rep.shape) == 3:
                rep = rep.mean(dim=2)  # 全局平均池化
            else:
                raise ValueError("Unexpected feature tensor shape")

            # 计算余弦相似度
            sim = cosine_similarity(input_emb, rep, dim=1)
            similarities.append((img_name, sim.item()))

        # 按相似度排序并选择前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        # 获取最相似图像的原始路径
        similar_images = [(os.path.join(self.img_path, img_name), sim) for img_name, sim in top_k]
        return similar_images
    
    def find_similar_images_str(self, input_img, lesion_str, k=2, layer=11):
        """
        对于输入图片，计算其与emb_folder中保存的所有嵌入的相似度，
        并返回最相似的前k个图像的原始路径。只包括详细信息中包含指定字符串的图片。
        
        :param input_img: 输入图片的路径
        :param lesion_str: 需要在详细信息中包含的字符串
        :param k: 返回最相似图像的数量，默认为5
        :param layer: 特征提取层的索引
        :return: 一个列表，包含最相似图像的原始路径和相似度分数
        """
        # 获取输入图片的嵌入
        input_emb = self.get_layer_representation(input_img, layer_index=layer)
        
        # 确保输入嵌入是一个二维张量 (batch_size, feature_dim)
        if len(input_emb.shape) == 4:
            input_emb = input_emb.mean(dim=(2, 3))  # 全局平均池化
        elif len(input_emb.shape) == 3:
            input_emb = input_emb.mean(dim=2)  # 全局平均池化
        else:
            raise ValueError("Unexpected feature tensor shape")

        # 加载文件夹中的所有嵌入
        representations = self.load_image_representations(self.emb_folder)

        # 过滤图片
        lesion_str = convert_abbreviation_to_full_name(lesion_str) 
        # print(lesion_str)
        filtered_images = self.filter_images_by_lesion(lesion_str)
        filtered_images = [os.path.basename(i) for i in filtered_images]
        # print(filtered_images)
        filtered_representations = {img_name: rep for img_name, rep in representations.items() if img_name in filtered_images}

        # 计算所有嵌入与输入嵌入的相似度
        similarities = []
        for img_name, rep in filtered_representations.items():
            # 确保预加载的嵌入也是一个二维张量 (batch_size, feature_dim)
            if len(rep.shape) == 4:
                rep = rep.mean(dim=(2, 3))  # 全局平均池化
            elif len(rep.shape) == 3:
                rep = rep.mean(dim=2)  # 全局平均池化
            else:
                raise ValueError("Unexpected feature tensor shape")

            # 计算余弦相似度
            sim = cosine_similarity(input_emb, rep, dim=1)
            similarities.append((img_name, sim.item()))
        # print(similarities)
        # 按相似度排序并选择前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        # 获取最相似图像的原始路径
        similar_images = [(os.path.join(self.img_path, img_name), sim) for img_name, sim in top_k]
        # print(similar_images)
        return similar_images
    
    def get_detailed_similarities(self, input_img, k=5, layer=11):
        """
        获取输入图片与文件夹中保存的嵌入的相似度，并返回最相似的前k个图像的详细信息。
        
        :param input_img: 输入图片的路径
        :param emb_folder: 包含预计算嵌入和对应关系的文件夹路径
        :param json_file: 包含图像详细信息的JSON文件路径
        :param k: 返回最相似图像的数量，默认为5
        :return: 一个列表，包含最相似图像的score, dis, 和 imid
        """
        # 获取最相似的图像
        similar_images = self.find_similar_images(input_img, k=k, layer=layer)
        # print(similar_images)
        # 读取JSON文件中的详细信息
        with open(self.json_file, 'r') as f:
            image_details = json.load(f)

        # 创建一个字典以便快速查找
        image_dict = {os.path.basename(detail['image_path']): detail for detail in image_details}

        # 获取详细的相似信息
        # detailed_similarities = 
        score_ = []
        txt_ = []
        metadata_ = []
        img_ = []
        for img_path, score in similar_images:
            img_name = os.path.basename(img_path)
            if img_name in image_dict:
                detail = image_dict[img_name]
                score_.append(score)
                txt_.append(detail["dis"])
                metadata_.append(detail["imid"])
                img_.append(img_path)
        detailed_similarities = {"score":score_, "txt": txt_, "metadata": metadata_, "img": img_}

        return detailed_similarities
    
class ClassicEmbBuilder(EmbBuilder):
    def __init__(self, img_folder, emb_folder):
        super().__init__(img_folder, emb_folder)
        
    def find_similar_images(self, input_img, k=2, layer=11):
        """
        对于输入图片，计算其与emb_folder中保存的所有嵌入的相似度，
        并返回最相似的前k个图像的原始路径。
        
        :param input_img: 输入图片的路径
        :param emb_folder: 包含预计算嵌入和对应关系的文件夹路径
        :param k: 返回最相似图像的数量，默认为5
        :return: 一个列表，包含最相似图像的原始路径和相似度分数
        """
        # 获取输入图片的嵌入
        input_emb = self.get_layer_representation(input_img, layer_index=layer)
        
        # 确保输入嵌入是一个二维张量 (batch_size, feature_dim)
        if len(input_emb.shape) == 4:
            input_emb = input_emb.mean(dim=(2, 3))  # 全局平均池化
        elif len(input_emb.shape) == 3:
            input_emb = input_emb.mean(dim=2)  # 全局平均池化
        else:
            raise ValueError("Unexpected feature tensor shape")

        # 加载文件夹中的所有嵌入
        representations = self.load_image_representations(self.emb_folder)

        # 计算所有嵌入与输入嵌入的相似度
        similarities = []
        for img_name, rep in representations.items():
            # 确保预加载的嵌入也是一个二维张量 (batch_size, feature_dim)
            if len(rep.shape) == 4:
                rep = rep.mean(dim=(2, 3))  # 全局平均池化
            elif len(rep.shape) == 3:
                rep = rep.mean(dim=2)  # 全局平均池化
            else:
                raise ValueError("Unexpected feature tensor shape")

            # 计算余弦相似度
            sim = cosine_similarity(input_emb, rep, dim=1)
            similarities.append((img_name, sim.item()))

        # 按相似度排序并选择前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        # 获取最相似图像的原始路径
        similar_images = [(os.path.join(self.img_path, img_name), sim) for img_name, sim in top_k]
        return similar_images
    
    def save_image_representation(self, source_folder, target_folder, layer_index=11):
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        image_data = self.data_extractor.extract_image_data_classic(source_folder)
        representation_data = {}
        for image_path, text, meta_data in image_data:
            file_path = os.path.join(source_folder, "/".join(image_path.split("/")[-2:]))
            representation = self.get_layer_representation(file_path, layer_index)
            
            # Save the representation as a .pt file
            representation_file = os.path.join(target_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}.pt")
            torch.save(representation, representation_file)

            # Record the correspondence
            representation_data[image_path] = representation_file

        # Save the correspondence data to a JSON file
        correspondence_file = os.path.join(target_folder, 'correspondence.json')
        with open(correspondence_file, 'w') as f:
            json.dump(representation_data, f)
            
    def get_detailed_similarities_crop(self, input_img, k=5):
        """
        获取输入图片与文件夹中保存的嵌入的相似度，并返回最相似的前k个图像的详细信息。
        
        :param input_img: 输入图片的路径
        :param emb_folder: 包含预计算嵌入和对应关系的文件夹路径
        :param json_file: 包含图像详细信息的JSON文件路径
        :param k: 返回最相似图像的数量，默认为5
        :return: 一个列表，包含最相似图像的score, dis, 和 imid
        """
        # 获取最相似的图像
        similar_images = self.find_similar_images(input_img, k=k)


        # 获取详细的相似信息
        score_ = []
        txt_ = []
        metadata_ = []
        img_ = []
        for img_path, score in similar_images:
            score_.append(score)
            txt_.append(img_path.split("/")[-1].split(".")[0])
            metadata_.append(img_path)
            img_.append("." + ".".join(img_path.split(".")[-2:]))
        detailed_similarities = {"score":score_, "txt": txt_, "metadata": metadata_, "img": img_}

        return detailed_similarities