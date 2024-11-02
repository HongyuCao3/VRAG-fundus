import torch
import os, json
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import argparse
from torch.nn.functional import cosine_similarity
from utils import find_json_file


class EmbBuilder():
    def __init__(self, img_folder, emb_folder):
        self.img_path = args.img_folder
        self.emb_folder = args.emb_path
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.json_file = find_json_file(self.img_path)
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
    
    def save_image_representations(self, source_folder, target_folder, layer_index=11):
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

    def load_image_representations(self, target_folder):
        correspondence_file = os.path.join(target_folder, 'correspondence.json')
        with open(correspondence_file, 'r') as f:
            representation_data = json.load(f)
        
        representations = {}
        for img_name, rep_path in representation_data.items():
            representation = torch.load(rep_path)
            representations[img_name] = representation
        
        return representations
    
    def find_similar_images(self, input_img, k=2):
        """
        对于输入图片，计算其与emb_folder中保存的所有嵌入的相似度，
        并返回最相似的前k个图像的原始路径。
        
        :param input_img: 输入图片的路径
        :param emb_folder: 包含预计算嵌入和对应关系的文件夹路径
        :param k: 返回最相似图像的数量，默认为5
        :return: 一个列表，包含最相似图像的原始路径和相似度分数
        """
        # 获取输入图片的嵌入
        input_emb = self.get_layer_representation(input_img)
        
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
        similar_images = [(os.path.join(self.emb_folder, img_name), sim) for img_name, sim in top_k]
        return similar_images
    
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

        # 读取JSON文件中的详细信息
        with open(self.json_file, 'r') as f:
            image_details = json.load(f)

        # 创建一个字典以便快速查找
        image_dict = {os.path.basename(detail['image_path']): detail for detail in image_details}

        # 获取详细的相似信息
        detailed_similarities = []
        for img_path, score in similar_images:
            img_name = os.path.basename(img_path)
            if img_name in image_dict:
                detail = image_dict[img_name]
                detailed_similarities.append({"score":score, "txt": detail['dis'], "metadata": detail['imid'], "img": img_path})

        return detailed_similarities

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 使用示例
    parser.add_argument("--img-path", type=str, default="./data/level")
    parser.add_argument("--emb-path", type=str, default="./data/level_emb_clip")
    args = parser.parse_args()
    EB = EmbBuilder(args.img_path, args.emb_path)
    img_path1 = './data/level/ODIR_2450_right.jpg'
    img_path2 = './data/level/ODIR_3259_left.jpg'
    # input_img = './data/DR/multidr/39dis_1ffa92f4-8d87-11e8-9daf-6045cb817f5b.jpg'
    input_img = './data/level/ODIR_2450_right.jpg'
    # sim = EB.calculate_similarity(img_path1, img_path2)
    # print(sim)
    # EB.save_image_representations(args.img_path, args.emb_path)
    print(EB.get_detailed_similarities(input_img))