import clip
import os, json
import torch
from PIL import Image
import numpy as np
import argparse

class ImgEmb():
    def __init__(self, device):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.device = device
    
    def get_features_from_image_path(self, image_paths):
        images = [self.preprocess(Image.open(image_path).convert("RGB")) for image_path in image_paths]
        image_input = torch.tensor(np.stack(images)).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
        return image_features
    
    def color_distance(self, color1, color2):
        return abs(int(color1[0]) - int(color2[0])) + abs(int(color1[1]) - int(color2[1])) + abs(int(color1[2]) - int(color2[2]))
    
    def get_lesion_colors(self, image_path):
        # 打开图片并转换为RGB格式
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # 获取所有颜色
        colors = img_array.reshape(-1, img_array.shape[-1])
        unique_colors = []
        
        for color in colors:
            # 检查颜色是否在unique_colors中
            if not any(self.color_distance(color, c) < 100 for c in unique_colors):
                unique_colors.append(color)

        return np.array(unique_colors)
        
    def open_images(self, image_path, segmentation_path):
        """打开眼底图和分割图并转换为RGB格式。"""
        image = Image.open(image_path).convert('RGB')
        segmentation = Image.open(segmentation_path).convert('RGB')
        return np.array(image), np.array(segmentation)

    def create_mask(self, segmentation_data, target_color, tolerance):
        """根据目标颜色和容差创建掩膜。"""
        lower_bound = np.array(target_color) - tolerance
        upper_bound = np.array(target_color) + tolerance
        mask = np.all((segmentation_data >= lower_bound) & (segmentation_data <= upper_bound), axis=-1)
        return mask

    def crop_and_save_lesions(self, image_data, segmentation_data, color_map, tolerance, output_dir):
        """根据分割图裁剪并保存病灶。"""
        os.makedirs(output_dir, exist_ok=True)

        for color, label in color_map.items():
            mask = self.create_mask(segmentation_data, color, tolerance)

            if np.any(mask):
                coords = np.argwhere(mask)
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0) + 1

                lesion_crop = image_data[y0:y1, x0:x1]
                cropped_image = Image.fromarray(lesion_crop)
                cropped_image.save(os.path.join(output_dir, f'{label}.png'))
                
    def create_color_images(self, colors, output_dir='output_colors', json_file='color_mapping.json'):
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        color_mapping = {}

        for i, color in enumerate(colors):
            # 创建纯色图像
            img = Image.new('RGB', (100, 100), tuple(color))  # 100x100的纯色图像
            file_name = f"color_{i+1}.png"
            img.save(f"{output_dir}/{file_name}")
            
            # 存储颜色和文件名的映射
            color_mapping[i + 1] = {
                'color': color.tolist(),
                'image': file_name
            }

        # 保存到JSON文件
        with open(json_file, 'w') as f:
            json.dump(color_mapping, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img-path", type=str, default="./segmentation/")
    parser.add_argument("--color-path", type=str, default="./segmentation/color.jpg")    
    args = parser.parse_args()
    image_paths = [os.path.join(args.img_path,file) for file in os.listdir(args.img_path)]
    IE = ImgEmb(args.device)
    colors = IE.get_lesion_colors(args.color_path)
    IE.create_color_images(colors, output_dir="./data/output_colors", json_file="./data/color_mapping.json")
    
    # print(image_paths)
    # feature = IE.get_features_from_image_path(image_paths)
    # print(feature)