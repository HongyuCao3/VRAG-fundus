import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import argparse
from torch.nn.functional import cosine_similarity


class EmbBuilder():
    def __init__(self, args,):
        pass
    def get_layer_representation(self, img_path, layer_index=11):
        # 加载预训练的CLIP模型和处理器
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 加载并预处理图像
        image = Image.open(img_path)
        inputs = processor(images=image, return_tensors="pt")

        # 获取模型的视觉编码器
        visual_model = model.vision_model

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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 使用示例
    args = parser.parse_args()
    EB = EmbBuilder(args)
    img_path1 = './data/level/ODIR_2450_right.jpg'
    img_path2 = './data/level/ODIR_3259_left.jpg'
    sim = EB.calculate_similarity(img_path1, img_path2)
    print(sim)