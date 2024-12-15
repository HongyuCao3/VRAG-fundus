import argparse
import torch
import os
import json
from tqdm import tqdm

from PIL import Image
import math
from transformers import set_seed, logging

from emb_builder import EmbBuilder
from emb_module.emb_builder import ClassicEmbBuilder
from context_former import ContextFormer
from utils import split_image, delete_images, merge_dicts

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
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
from VRAG_Framework.vrag_filter import VRAGFilter
from VRAG_Framework.checker import Checker

class VRAG():
    def __init__(self, args):
        self.model_path = args.model_path
        self.top_k_c = args.top_k_c # forcrop emb
        self.top_k_l = args.top_k_l # for level emb
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, args.model_base, model_name
        )
        self.conv_mode = args.conv_mode
        self.temperature = args.temperature
        self.top_p=args.top_p
        self.num_beams=args.num_beams
        self.image_folder = args.image_folder
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
        self.classic_emb_path = args.classic_emb_path
        self.layer = args.layer
        self.context_former = ContextFormer(args.use_pics)
        self.filter = args.filter
        self.check = args.check
        self.t_check = args.t_check
        self.t_filter = args.t_filter
        self.load_embs()
        self.vrag_filter = VRAGFilter(self.context_former, threshold=self.t_filter)
        self.checker = Checker(threshold=self.t_check)
    
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
            
        
    def inference_rag(self, query_str, img_path):
        # do retrieval
        if self.chunk_m == 1 and self.chunk_n == 1:
            ret_c, ret_l= self.retrieve(img_path)
        else:
            sub_imgs = split_image(img_path, self.tmp_path, self.chunk_m, self.chunk_n)
            ret_cs = []
            for sub_img in sub_imgs:
                ret_c, ret_l= self.retrieve(sub_img)
                ret_cs.append(ret_c)
            ret_c = merge_dicts(ret_cs)
        
        # form context
        prompt, images, record_data = self.form_context(img_path, query_str, ret_c, ret_l)
            
        # do inference
        set_seed(0)
        disable_torch_init()
        qs = prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_tensor = process_images(images, self.image_processor, self.model.config)[0] 
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # record_data.update({"outputs": outputs})
        # delete_images()
        return outputs, record_data
    
    def inference_rag_all(self, query_str, img_path):
        # do retrieval
        if self.classic_emb == None:
            ret_cl = self.context_former.ret_empty
        else:
            ret_cl = self.classic_emb.get_detailed_similarities_crop(img_path, 1)
            
        ret_cl_ = ret_cl
        if self.filter:
            ret_cl = self.vrag_filter.filter_multi_modal_vqa(ret_cl)
        # form context
        prompt, images, record_data = self.context_former.form_context_all_cl(img_path, query_str, ret_cl)
            
        # do inference
        # set_seed(0)
        # disable_torch_init()
        # qs = prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        # if self.model.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        # conv = conv_templates[self.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        
        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # image_tensor = process_images(images, self.image_processor, self.model.config)[0] 
        
        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        # with torch.inference_mode():
        #         output_ids = self.model.generate(
        #             input_ids,
        #             images=image_tensor.unsqueeze(0).half().cuda(),
        #             do_sample=True if self.temperature > 0 else False,
        #             temperature=self.temperature,
        #             top_p=self.top_p,
        #             num_beams=self.num_beams,
        #             # no_repeat_ngram_size=3,
        #             max_new_tokens=1024,
        #             use_cache=True)

        # outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # delete_images()
        outputs = self.model_chat(images, prompt)
        record_data.update({"outputs": outputs})
        if self.check:
            flag, check_str = self.checker.check_multi_modal_vqa(outputs, ret_cl_)
            if not flag:
                outputs = self.model_chat(images, check_str+prompt)
        return outputs, record_data
    
    def model_chat(self, images, prompt):
        set_seed(0)
        disable_torch_init()
        qs = prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_tensor = process_images(images, self.image_processor, self.model.config)[0] 
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def form_context(self, img_path, query_str, ret_c, ret_l):
        record_data = {}
        record_data.update({"ret_c": str(ret_c)})
        record_data.update({"ret_l": str(ret_l)})
        record_data.update({"org": img_path})
            # img, txt, score, metadata = node
        # txt2img retrieve
        # img, txt, score, metadata = retrieve_data.text_to_image_retrieve(img_path)
        # print(score)
        image_documents = [ImageDocument(image_path=img_path)]
        image_org = Image.open(img_path)
        images= [image_org]
        img = ret_c["img"]
        img.extend(ret_l["img"])
        if self.use_pics:
            for res_img in img:
                image_documents.append(ImageDocument(image_path=res_img))
                # print(res_img)
                image = Image.open(res_img)
                images.append(image)
        result_dict_c = dict(zip(ret_c["txt"], ret_c["score"]))
        result_dict_l = dict(zip(ret_l["txt"], ret_l["score"]))
        context_str_c = str(result_dict_c)
        context_str_l = str(result_dict_l)
        metadata_str = ret_c["metadata"]
        metadata_str.extend(ret_l["metadata"])
        prompt = self.build_diagnosis_string(context_str_l, context_str_c, "", self.diagnosis_str, metadata_str, query_str)
        record_data.update({"prompt": prompt})
        return prompt, images, record_data
            
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
        
    def build_diagnosis_string(self, context_str_l, context_str_c, context_str_cl, diagnosis_str, metadata_str, query_str):
        parts = []
        
        if diagnosis_str != "{}":
            parts.append(f"Diagnosing Standard: {diagnosis_str}\n")
        if context_str_l != "{}":
            parts.append(f"The possible diagnosing level and probability: {context_str_l}\n")
        if context_str_c != "{}":
            parts.append(f"The possible lesion and probability: {context_str_c}\n")
        if context_str_cl != "{}":
            parts.append(f"The possible diagnosing class and probability: {context_str_cl}\n")
        # if metadata_str != "[{}]":
        #     parts.append(f"Metadata: {metadata_str}\n")
        
        parts.append("---------------------\n")
        parts.append(f"Query: {query_str}\n")
        parts.append("Answer: ")
        
        return "".join(parts)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--image-folder", type=str, default="./segmentation/")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--meta-data", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/data/segmentation.json")
    parser.add_argument("--chunk-m", type=int, default=1)
    parser.add_argument("--chunk-n", type=int, default=1)
    parser.add_argument("--tmp-path", type=str, default="./data/tmp")
    parser.add_argument("--emb-path", type=str, default="./data/emb_crop")
    parser.add_argument("--layer", type=int, default=11)
    args = parser.parse_args()
    vrag = VRAG(args)
    # print(vrag.inference())
    # vrag.build_index()
    test_img = "/home/hongyu/DDR/lesion_segmentation/test/image/007-1789-100.jpg"
    query_str_0 = "Can you describe the image in details?"
    query_str_1 = "what's the diagnosis?"
    print(vrag.inference_rag(query_str_1, test_img))
    # vrag.build_index()