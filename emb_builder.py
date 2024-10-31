import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import argparse

class EmbBuilder():
    def __init__(self):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 使用示例
    args = parser.parse_args()
    EB = EmbBuilder(args)
    img_path = './data/level/ODIR_2450_right.jpg'
    layer_representation = EB.get_layer_representation(img_path, layer_index=11)
    print(layer_representation.shape)  # 打印输出张量的形状