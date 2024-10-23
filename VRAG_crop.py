import argparse
import torch
import os
import json
from tqdm import tqdm

from PIL import Image
import math
from transformers import set_seed, logging

from utils import split_image, delete_images

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

class VRAG():
    def __init__(self, args):
        self.model_path = args.model_path
        self.top_k_c = args.top_k_c # forcrop emb
        self.top_k_l = args.top_k_l # for level emb
        self.top_k_cl = args.top_k_cl # for level emb
        self.use_pics = args.use_pics
        self.use_rag = args.use_rag
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, args.model_base, model_name
        )
        self.conv_mode = args.conv_mode
        self.temperature = args.temperature
        self.top_p=args.top_p
        self.num_beams=args.num_beams
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.image_folder = args.image_folder
        diagnosing_level = {"Mild NPDR": "MAs only", "Moderate NPDR": "At least one hemorrhage or MA and/or at least one of the following: Retinal hemorrhages, Hard exudates, Cotton wool spots, Venous beading", "Severe NPDR": "Any of the following but no signs of PDR (4-2-1 rule): >20 intraretinal hemorrhages in each of four quadrants, definite venous, beading in two or more quadrants, Prominent IRMA in one or more quadrants", "PDR": "One of either: Neovascularization, Vitreous/preretinal hemorrhage"}
        self.diagnosis_str = ""
        for key, value in diagnosing_level.items():
            self.diagnosis_str += f"{key}: {value}"
        self.chunk_m = args.chunk_m
        self.chunk_n = args.chunk_n
        self.tmp_path = args.tmp_path
        self.crop_emb_path = args.crop_emb_path
        self.level_emb_path = args.level_emb_path
        self.classic_emb_path = args.classic_emb_path
        self.load_embs()
    
    def load_embs(self, ):
        self.crop_multi_index = self.load_emb(self.crop_emb_path)
        self.level_multi_index = self.load_emb(self.level_emb_path)
        self.classic_multi_index = self.load_emb(self.classic_emb_path)
            
    def load_emb(self, emb_path):
        if emb_path != None:
            if os.path.exists(emb_path):
                storage_context_classic = StorageContext.from_defaults(persist_dir=emb_path)
                multi_index = load_index_from_storage(storage_context_classic)
            else:
                print("invalid emb "+emb_path)
                multi_index=None
        else:
            print("None emb")
            multi_index=None
        return multi_index
        
    def inference_rag(self, query_str, img_path):
        # do retrieval
        if self.chunk_m == 1 and self.chunk_n == 1:
            ret_c, ret_l, ret_cl = self.retrieve(img_path)
        else:
            sub_imgs = split_image(img_path, self.tmp_path, self.chunk_m, self.chunk_n)
            for sub_img in sub_imgs:
                ret_c, ret_l = self.retrieve(sub_img)
                # TODO：添加计数方式
            # TODO：删除临时图片
        
        # form context
        prompt, images, record_data = self.form_context(img_path, query_str, ret_c, ret_l, ret_cl)
            
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
        return outputs, record_data
    
    def form_context(self, img_path, query_str, ret_c, ret_l, ret_cl):
        record_data = {}
        record_data.update({"ret_c": str(ret_c)})
        record_data.update({"ret_l": str(ret_l)})
        record_data.update({"ret_cl": str(ret_cl)})
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
        result_dict_cl = dict(zip(ret_cl["txt"], ret_cl["score"]))
        context_str_c = str(result_dict_c)
        context_str_l = str(result_dict_l)
        context_str_cl = str(result_dict_cl)
        metadata_str = ret_c["metadata"]
        metadata_str.extend(ret_l["metadata"])
        metadata_str.extend(ret_cl["metadata"])
        prompt = self.build_diagnosis_string(context_str_l, context_str_c, context_str_cl, self.diagnosis_str, metadata_str, query_str)
        record_data.update({"prompt": prompt})
        return prompt, images, record_data
            
    def retrieve(self, img_path):
        crop_ret = self.retrieve_from_emb(self.crop_multi_index, img_path, self.top_k_c)
        level_ret = self.retrieve_from_emb(self.level_multi_index, img_path, self.top_k_l)
        classic_ret = self.retrieve_from_emb(self.classic_multi_index, img_path, self.top_k_cl)
                
        return crop_ret, level_ret, classic_ret
    
    def retrieve_from_emb(self, multi_index, img_path, top_k):
        txt = []
        score = [] 
        img = [] 
        metadata = []
        if multi_index != None:
            retrieve_data = multi_index.as_retriever(similarity_top_k=top_k, image_similarity_top_k=top_k)
            nodes = retrieve_data.image_to_image_retrieve(img_path)
            for node in nodes:
                txt.append(node.get_text()) # excudates
                score.append(node.get_score()) # 0.628
                img.append(node.node.image_path)
                metadata.append(node.node.metadata)
        return {"txt": txt, "score": score, "img": img, "metadata": metadata}
    
        
    def build_diagnosis_string(self, context_str_l, context_str_c, context_str_cl, diagnosis_str, metadata_str, query_str):
        parts = []
        
        if context_str_l != "{}":
            parts.append(f"The possible diagnosing level and probability: {context_str_l}\n")
        if context_str_c != "{}":
            parts.append(f"The possible lesion and probability: {context_str_c}\n")
        if context_str_cl != "{}":
            parts.append(f"The possible diagnosing class and probability: {context_str_cl}\n")
        if diagnosis_str != "{}":
            parts.append(f"Diagnosing Standard: {diagnosis_str}\n")
        if metadata_str != "[{}]":
            parts.append(f"Metadata: {metadata_str}\n")
        
        parts.append("---------------------\n")
        parts.append(f"Query: {query_str}\n")
        parts.append("Answer: ")
        
        return "".join(parts)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/data/eye_diag.json")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--image-folder", type=str, default="./segmentation/")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--meta-data", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/data/segmentation.json")
    parser.add_argument("--use-pics", type=bool, default=False)
    parser.add_argument("--use-rag", type=bool, default=False)
    parser.add_argument("--chunk-m", type=int, default=1)
    parser.add_argument("--chunk-n", type=int, default=1)
    parser.add_argument("--tmp-path", type=str, default="./data/tmp")
    parser.add_argument("--emb-path", type=str, default="./data/emb_crop")
    args = parser.parse_args()
    vrag = VRAG(args)
    # print(vrag.inference())
    # vrag.build_index()
    test_img = "/home/hongyu/DDR/lesion_segmentation/test/image/007-1789-100.jpg"
    query_str_0 = "Can you describe the image in details?"
    query_str_1 = "what's the diagnosis?"
    print(vrag.inference_rag(query_str_1, test_img))
    # vrag.build_index()