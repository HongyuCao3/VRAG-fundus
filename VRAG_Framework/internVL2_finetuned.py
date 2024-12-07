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
from emb_module.emb_builder import ClassicEmbBuilder
# from InternVL.internvl_chat.internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from context_former import ContextFormer
from utils import split_image, delete_images, merge_dicts, find_longest_diagnosis_keys, expand_disease_abbreviation
from internvl.model import load_model_and_tokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
load_8bit=True

class InternVL2_finetuned():
    def __init__(self, args, ):
        # self.model = AutoModel.from_pretrained(
        #     args.model_path,
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        #     use_flash_attn=True,
        #     device_map={"":0},
        #     trust_remote_code=True).eval().cuda()
        # self.model = InternVLChatModel.from_pretrained(
        #     args.model_path,
        #     load_in_8bit=load_8bit,
        #     torch_dtype=torch.bfloat16,
        #     device_map='auto').eval()
        # self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
        self.model, self.tokenizer = load_model_and_tokenizer(args.model_path) 
        self.top_k_c = args.top_k_c # forcrop emb
        self.top_k_l = args.top_k_l # for level emb
        self.chunk_m = args.chunk_m
        self.chunk_n = args.chunk_n
        self.tmp_path = args.tmp_path
        self.use_pics = args.use_pics
        self.crop_emb_path = args.crop_emb_path
        self.level_emb_path = args.level_emb_path
        self.classic_emb_path = args.classic_emb_path
        self.layer = args.layer
        self.load_embs()
        self.context_former = ContextFormer(args.use_pics)
    
    def load_embs(self, ):
        if self.level_emb_path:
            self.level_emb = EmbBuilder("./data/level/", self.level_emb_path)
        else:
            self.level_emb = None
        if self.crop_emb_path:
            self.crop_emb = EmbBuilder("./data/lesion/", self.crop_emb_path)
        else:
            self.crop_emb = None
        if self.classic_emb_path:
            self.classic_emb = ClassicEmbBuilder("./data/Classic Images/", self.classic_emb_path)
        else:
            self.classic_emb = None
            
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
    
    def inference_rag(self, query_str, image_path ):
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
    
    def inference_multi_turn(self, query_str, image_path):
        record_data_f = {}
        # 第一轮要求给出基本诊断
        pixel_values = self.load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        # prompt, images, record_data = self.context_former.form_context(image_path, query_str, self.context_former.ret_empty, self.context_former.ret_empty)
        prompt = self.context_former.form_context_internvl_step1(image_path, query_str)
        question = prompt
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
        # print("Step 1: ", end="")
        # print(response)
        record_data_f.update({"step 1": {"response": response}})
        
        # 第二轮根据上次指出的病灶给出最相似的参考图要求做出轨迹和颜色判断
        keys = find_longest_diagnosis_keys(response, self.context_former.lesion)
        print("find lesion key word", end="")
        print(keys)
        for key in keys:
            ret_c = self.crop_emb.get_detailed_similarities_str_crop(image_path, lesion_str=key, k=1)
            # TODO:需要整合ret_c
        if len(keys) == 0:
            pixel_values_c = pixel_values
            ret_c = self.context_former.ret_empty
        else:
            pixel_values_c = self.load_image(ret_c["img"][0], max_num=12).to(torch.bfloat16).cuda()
            pixel_values_c = torch.cat((pixel_values, pixel_values_c), dim=0)
        prompt, images, record_data = self.context_former.form_context_c(image_path, query_str, ret_c)
        question = prompt
        response, history = self.model.chat(self.tokenizer, pixel_values_c, question, generation_config, history=history, return_history=True)
        # print("Step 2: ", end="")
        # print(response)
        record_data_f.update({"step 2": {"response": response, "record_data": record_data}})
        
        # 第三轮根据基本诊断的几种可能给出最相似的参考图要求做出多图推理
        keys = find_longest_diagnosis_keys(response, self.context_former.diagnosing_level)
        print("find level key word", end="")
        print(keys)
        for key in keys:
            ret_l = self.level_emb.get_detailed_similarities_str(image_path, lesion_str=key, k=1)
            # TODO:需要整合ret_l
        if len(keys) == 0:
            pixel_values_l = pixel_values
            ret_c = self.context_former.ret_empty
        else:
            pixel_values_l = self.load_image(ret_l["img"][0], max_num=12).to(torch.bfloat16).cuda()
            pixel_values_l = torch.cat((pixel_values, pixel_values_l), dim=0)
            prompt, images, record_data = self.context_former.form_context_l(image_path, query_str, ret_l)
        question = prompt
        response, history = self.model.chat(self.tokenizer, pixel_values_l, question, generation_config, history=history, return_history=True)
        # print("Step 3: ", end="")
        # print(response)
        record_data_f.update({"step 3": {"response": response, "record_data": record_data}})
        
        # 第四轮要求根据之前的分析给出最终诊疗结果
        prompt, images, record_data = self.context_former.form_context_all(image_path, query_str,)
        question = prompt
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
        # print("Step 4: ", end="")
        # print(response)
        record_data_f.update({"step 4": {"response": response, "record_data": record_data}})
        return response, record_data_f
    
    def inference_multi_turn_check(self, query_str, image_path):
        record_data_f = {}
        # 第一轮要求给出基本诊断
        pixel_values = self.load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        prompt = self.context_former.form_context_internvl_step1(image_path, query_str)
        question = prompt
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
        record_data_f.update({"step 1": {"response": response}})
        
        
        # 第二轮根据上一轮的答案对比kb匹配结果，要求进一步解释
        keys = find_longest_diagnosis_keys(response, self.context_former.diagnosing_level)
        ret_l = self.level_emb.get_detailed_similarities(image_path, k=1)
        pixel_values_l = self.load_image(ret_l["img"][0], max_num=12).to(torch.bfloat16).cuda()
        pixel_values_l = torch.cat((pixel_values, pixel_values_l), dim=0)
        prompt, images, record_data = self.context_former.form_context_l_check(image_path, query_str, ret_l, keys)
        question = prompt
        response, history = self.model.chat(self.tokenizer, pixel_values_l, question, generation_config, history=history, return_history=True)
        # print("Step 3: ", end="")
        # print(response)
        record_data_f.update({"step 2": {"response": response, "record_data": record_data}})
        
        # 第四轮要求根据之前的分析给出最终诊疗结果
        prompt, images, record_data = self.context_former.form_context_all(image_path, query_str,)
        question = prompt
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
        # print("Step 4: ", end="")
        # print(response)
        record_data_f.update({"step 3": {"response": response, "record_data": record_data}})
        return response, record_data_f
    
    def inference_rag_all(self, query_str, img_path):
        # do retrieval
        if self.classic_emb == None:
            ret_cl = self.context_former.ret_empty
        else:
            ret_cl = self.classic_emb.get_detailed_similarities_crop(img_path, 1)
        # form context
        prompt, images, record_data = self.context_former.form_context_all_cl(img_path, query_str, ret_cl)
            
        # do inference
        pixel_values = self.load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
        # TODO:需要考虑输入多张图片
        # pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        # generation_config = dict(max_new_tokens=1024, do_sample=False)
        generation_config = dict(
                num_beams=args.num_beams,
                # max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
        )
        # single-image single-round conversation (单图单轮对话)
        question = prompt
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config, verbose=True)
        # print(f'User: {question}\nAssistant: {response}')
        return response, record_data
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--model-path", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--top-k-c", type=int, default=3)
    parser.add_argument("--top-k-l", type=int, default=1)
    parser.add_argument("--chunk-m", type=int, default=1)
    parser.add_argument("--chunk-n", type=int, default=1)
    parser.add_argument("--tmp-path", type=str, default="./data/tmp")
    parser.add_argument("--dataset", type=str, default="DR")
    parser.add_argument("--query-str", type=str, default="what's the diagnosis level?")
    parser.add_argument("--crop-emb-path", type=str, default=None)
    parser.add_argument("--level-emb-path", type=str, default=None)
    parser.add_argument("--use-pics", type=bool, default=False)
    args = parser.parse_args()
    test_img = "/home/hongyu/DDR/lesion_segmentation/test/image/007-1789-100.jpg"
    IV2 = InternVL2_finetuned(args)
    print(IV2.inference_multi_turn(args.query_str, test_img))