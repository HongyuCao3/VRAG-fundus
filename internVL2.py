import argparse
import os
import json
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from emb_builder import EmbBuilder
from context_former import ContextFormer
from utils import split_image, delete_images, merge_dicts


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVL2():
    def __init__(self, model_path= 'OpenGVLab/InternVL2-8B', ):
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.top_k_c = args.top_k_c # forcrop emb
        self.top_k_l = args.top_k_l # for level emb
        diagnosing_level = {"Normal": "No lesion","Mild NPDR": "MAs only", "Moderate NPDR": "At least one hemorrhage or MA and/or at least one of the following: Retinal hemorrhages, Hard exudates, Cotton wool spots, Venous beading", "Severe NPDR": "Any of the following but no signs of PDR (4-2-1 rule): >20 intraretinal hemorrhages in each of four quadrants, definite venous, beading in two or more quadrants, Prominent IRMA in one or more quadrants", "PDR": "One of either: Neovascularization, Vitreous/preretinal hemorrhage"}
        self.diagnosis_str = ""
        for key, value in diagnosing_level.items():
            self.diagnosis_str += f"{key}: {value}"
        self.chunk_m = args.chunk_m
        self.chunk_n = args.chunk_n
        self.tmp_path = args.tmp_path
        self.use_pics = args.use_pics
        self.crop_emb_path = args.crop_emb_path
        self.level_emb_path = args.level_emb_path
        self.layer = args.layer
        self.load_embs()
        self.context_former = ContextFormer()
    
    def load_embs(self, ):
        if self.level_emb_path:
            self.level_emb = EmbBuilder("./data/level/", self.level_emb_path)
        else:
            self.level_emb = None
        if self.crop_emb_path:
            self.crop_emb = EmbBuilder("./data/lesion/", self.crop_emb_path)
        else:
            self.crop_emb = None
            
    def retrieve(self, img_path):
        ret_empty = {"img": [], "txt": [], "score": [], "metadata": []}
        if self.crop_emb:
            ret_c = self.crop_emb.get_detailed_similarities_crop(img_path, self.top_k_c)
        else:
            ret_c = ret_empty
        if self.level_emb:
            ret_l = self.level_emb.get_detailed_similarities(img_path, self.top_k_l, self.layer)
        else:
            ret_l = ret_empty
        return ret_c, ret_l
    
    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def inference(self, query_str, image_path ):
        # retrivev and form context
        if self.chunk_m == 1 and self.chunk_n == 1:
            ret_c, ret_l= self.retrieve(image_path)
        else:
            sub_imgs = split_image(image_path, self.tmp_path, self.chunk_m, self.chunk_n)
            ret_cs = []
            for sub_img in sub_imgs:
                ret_c, ret_l= self.retrieve(sub_img)
                ret_cs.append(ret_c)
            ret_c = merge_dicts(ret_cs)
        prompt, images, record_data = self.context_former.form_context(image_path, query_str, ret_c, ret_l)
        # set the max number of tiles in `max_num`
        pixel_values = self.load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        # TODO:需要考虑输入多张图片
        # pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        # single-image single-round conversation (单图单轮对话)
        question = prompt
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        # print(f'User: {question}\nAssistant: {response}')
        return response, record_data
    
    def inference_mulit_turn(self, ):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--model-path", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B")
    parser.add_argument("--crop-emb-path", type=str, default=None)
    parser.add_argument("--level-emb-path", type=str, default=None)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--dataset", type=str, default="DR")
    parser.add_argument("--query-str", type=str, default="what's the diagnosis level?")
    args = parser.parse_args()
    test_img = "/home/hongyu/DDR/lesion_segmentation/test/image/007-1789-100.jpg"
    IV2 = InternVL2(model_path=args.model_path)
    print(IV2.inference(test_img, args.query_str))