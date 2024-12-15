import clip
import os, json
import torch
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

class ImgEmb():
    def __init__(self, color_map, device):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.device = device
        # if os.path.exists(color_map):
        with open(color_map, "r") as cf:
            self.color_map = json.load(cf)
        
    
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
        
    def extract_lesions(self, image_path, mask_path, output_dir='lesion_images', json_file='lesion_mapping.json', delta=100):
         # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 打开原图和分割图
        p = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        lesion_mapping = {}

        for lesion_name, color in self.color_map.items():
            # 创建掩模
            mask_array = np.array(mask)
            p_array = np.array(p)
            mask_mask = np.zeros(mask_array.shape[:2], dtype=bool)

            # 检查每个像素的颜色距离
            for i in range(mask_array.shape[0]):
                for j in range(mask_array.shape[1]):
                    if self.color_distance(mask_array[i, j], color) < delta:
                        mask_mask[i, j] = True

            # 提取病灶区域
            lesion_image = np.zeros_like(p_array)
            lesion_image[mask_mask] = p_array[mask_mask]

            # 创建病灶图像并保存
            lesion_img = Image.fromarray(lesion_image)
            file_name = f"{lesion_name}.png"
            lesion_img.save(f"{output_dir}/{file_name}")

            # 存储病灶名称和文件名的映射
            # print(image_path)
            lesion_mapping[lesion_name] = image_path.split("/")[-1].split(".")[0] +"/"+ file_name

        # 保存到JSON文件
        with open(json_file, 'w') as f:
            json.dump(lesion_mapping, f, indent=4)
    
    def get_images_segs(self, image_folder):
        # 创建一个字典来存储对应关系
        mapping = {}

        # 列出文件夹中的所有文件
        files = os.listdir(image_folder)

        # 遍历所有文件
        for filename in files:
            # 检查是否是png文件
            if filename.endswith('.png'):
                # 获取对应的jpg文件名
                seg_filename = filename.replace('.png', '.jpg')
                
                # 检查对应的jpg文件是否存在
                if seg_filename in files:
                    # 添加到字典中
                    mapping[filename] = seg_filename

        return mapping
    
    def extract_lesion_all(self, image_folder):
        images = self.get_images_segs(image_folder)
        for k, v in tqdm(images.items()):
            file = k.split(".")[0]
            mask_path = "./segmentation/{file}.jpg".format(file=file)
            img_path = "./segmentation/{file}.png".format(file=file)
            output_dir = "./data/lesion/{file}/".format(file=file)
            json_file="./data/lesion/lesion_map_{file}.json".format(file=file)
            # print(mask_path)
            # print(img_path)
            # print(output_dir)
            # print(json_file)
            self.extract_lesions(img_path ,mask_path, output_dir, json_file)
            
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img-path", type=str, default="./segmentation/")
    parser.add_argument("--color-path", type=str, default="./segmentation/color.jpg")    
    parser.add_argument("--color-map", type=str, default="./data/color.json")
    args = parser.parse_args()
    image_paths = [os.path.join(args.img_path,file) for file in os.listdir(args.img_path)]
    IE = ImgEmb(args.color_map, args.device)
    
    # 获取病灶颜色对应关系
    # colors = IE.get_lesion_colors(args.color_path)
    # IE.create_color_images(colors, output_dir="./data/output_colors", json_file="./data/color_mapping.json")
    
    # 根据seg提取病灶crop
    # file = "IDRiD_49"
    # mask_path = "./segmentation/{file}.jpg".format(file=file)
    # img_path = "./segmentation/{file}.png".format(file=file)
    # IE.extract_lesions(img_path ,mask_path, output_dir="./data/lesion/{file}/".format(file=file), json_file="./data/lesion/lesion_map_{file}.json".format(file=file))
    
    # 提取所有病灶
    IE.extract_lesion_all(args.img_path)