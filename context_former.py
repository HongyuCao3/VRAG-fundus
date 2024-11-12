import argparse
import torch
import os
import json
from tqdm import tqdm

from PIL import Image
from llama_index.core.schema import ImageDocument
from utils import convert_abbreviation_to_full_name, convert_full_name_to_abbreviation

class ContextFormer():
    def __init__(self, use_pics):
        self.use_pics = use_pics
        self.diagnosing_level = {"Normal": "No lesion","Mild NPDR": "MAs only", "Moderate NPDR": "At least one hemorrhage or MA and/or at least one of the following: Retinal hemorrhages, Hard exudates, Cotton wool spots, Venous beading", "Severe NPDR": "Any of the following but no signs of PDR (4-2-1 rule): >20 intraretinal hemorrhages in each of four quadrants, definite venous, beading in two or more quadrants, Prominent IRMA in one or more quadrants", "PDR": "One of either: Neovascularization, Vitreous/preretinal hemorrhage"}
        self.diagnosis_str = ""
        for key, value in self.diagnosing_level.items():
            self.diagnosis_str += f"{key}: {value}"
        self.ret_empty = {"img": [], "txt": [], "score": [], "metadata": []}
        self.lesion = {"microaneurysm": "", "hemorrhage": "", "cotton wool spots": "", "exudates": ""}
    
    def form_context_all_cl(self, img_path, query_str, ret_cl):
        record_data = {}
        record_data.update({"ret_cl": str(ret_cl)})
        record_data.update({"org": img_path})
            # img, txt, score, metadata = node
        # txt2img retrieve
        # img, txt, score, metadata = retrieve_data.text_to_image_retrieve(img_path)
        # print(score)
        image_documents = [ImageDocument(image_path=img_path)]
        image_org = Image.open(img_path)
        images= [image_org]
        img = ret_cl["img"]
        if self.use_pics:
            for res_img in img:
                image_documents.append(ImageDocument(image_path=res_img))
                # print(res_img)
                image = Image.open(res_img)
                images.append(image)
        result_dict_cl = dict(zip(ret_cl["txt"], ret_cl["score"]))
        context_str_cl = str(result_dict_cl)
        metadata_str = ret_cl["metadata"]
        metadata_str.extend(ret_cl["metadata"])
        prompt = self.build_diagnosis_string_all("", "", context_str_cl, "", metadata_str, query_str)
        record_data.update({"prompt": prompt})
        return prompt, images, record_data
    
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
    
    def form_context_internvl_step1(self, img_path, query_str):
        parts = []
        
        if self.diagnosis_str != "{}":
            parts.append(f"Diagnosing Standard: {self.diagnosis_str}\n")
        
        parts.append("---------------------\n")
        parts.append(f"Query: {query_str}\n")
        parts.append("Give the answer in format {\"level\": "", \"reasons\": ""}")
        parts.append("Answer: ")
        
        prompt = "".join(parts)
        return prompt
    
    def form_context_c(self, img_path, query_str, ret_c):
        record_data = {}
        record_data.update({"ret_c": str(ret_c)})
        record_data.update({"org": img_path})
        image_documents = [ImageDocument(image_path=img_path)]
        image_org = Image.open(img_path)
        images= [image_org]
        img = ret_c["img"]
        if self.use_pics:
            for res_img in img:
                image_documents.append(ImageDocument(image_path=res_img))
                # print(res_img)
                image = Image.open(res_img)
                images.append(image)
        result_dict_c = dict(zip(ret_c["txt"], ret_c["score"]))
        context_str_c = str(result_dict_c)
        metadata_str = ret_c["metadata"]
        prompt = "The second picture is the possible lesion " + context_str_c + " Please check the diagnosis again."
        # prompt = self.build_diagnosis_string("", context_str_c, "", self.diagnosis_str, metadata_str, query_str)
        record_data.update({"prompt": prompt})
        return prompt, images, record_data
    
    def form_context_l(self, img_path, query_str, ret_l):
        record_data = {}
        record_data.update({"ret_c": str(ret_l)})
        record_data.update({"org": img_path})
        image_documents = [ImageDocument(image_path=img_path)]
        image_org = Image.open(img_path)
        images= [image_org]
        img = ret_l["img"]
        if self.use_pics:
            for res_img in img:
                image_documents.append(ImageDocument(image_path=res_img))
                # print(res_img)
                image = Image.open(res_img)
                images.append(image)
        result_dict_l = dict(zip(ret_l["txt"], ret_l["score"]))
        context_str_l = str(result_dict_l)
        metadata_str = ret_l["metadata"]
        prompt = "The second picture is an example of the diagnosis level you have made " + context_str_l + " Please check the diagnosis again."
        # prompt = self.build_diagnosis_string("", context_str_c, "", self.diagnosis_str, metadata_str, query_str)
        record_data.update({"prompt": prompt})
        return prompt, images, record_data
    
    def form_context_all(self, img_path, query_str):
        record_data = {}
        image_org = Image.open(img_path)
        images = [image_org]
        prompt = "According to previous chat history, " + query_str
        return prompt, images, record_data
        
    
    def build_diagnosis_string(self, context_str_l, context_str_c, context_str_cl, diagnosis_str, metadata_str, query_str):
        parts = []
        
        if diagnosis_str != "{}":
            parts.append(f"Diagnosing Standard: {diagnosis_str}\n")
        if context_str_l != "{}":
            parts.append(f"The possible diagnosing level and probability: {context_str_l}\n")
        if context_str_c != "{}":
            parts.append(f"The possible lesion and probability: {context_str_c}\n")
        # if context_str_cl != "{}":
        #     parts.append(f"The possible diagnosing class and probability: {context_str_cl}\n")
        # if metadata_str != "[{}]":
        #     parts.append(f"Metadata: {metadata_str}\n")
        parts.append("Give the answer in format {\"level\": "", \"reasons\": ""}")
        parts.append("---------------------\n")
        parts.append(f"Query: {query_str}\n")
        parts.append("Answer: ")
        
        return "".join(parts)
    
    def build_diagnosis_string_all(self, context_str_l, context_str_c, context_str_cl, diagnosis_str, metadata_str_all, query_str):
        parts = []
        
        if context_str_cl != "{}":
            parts.append(f"The possible diagnosis and probability: {context_str_cl}\n")
        # if context_str_c != "{}":
            # parts.append(f"The possible lesion and probability: {context_str_c}\n")
        # if context_str_cl != "{}":
        #     parts.append(f"The possible diagnosing class and probability: {context_str_cl}\n")
        # if metadata_str != "[{}]":
        #     parts.append(f"Metadata: {metadata_str}\n")
        parts.append("Give the answer in format {\"diagnosis\": "", \"reasons\": ""}")
        parts.append("---------------------\n")
        parts.append(f"Query: {query_str}\n")
        parts.append("Answer: ")
        
        return "".join(parts)
    
    def form_context_l_check(self, img_path, query_str, ret_l, keys):
        record_data = {}
        record_data.update({"ret_c": str(ret_l)})
        record_data.update({"org": img_path})
        result_dict_l = dict(zip(ret_l["txt"], ret_l["score"]))
        context_str_l = str(result_dict_l)
        metadata_str = ret_l["metadata"]
        if ret_l["txt"][0] == keys[0] or ret_l["txt"][0] == convert_abbreviation_to_full_name(keys[0]):
            prefix = "Your diagnosis may be accurate"
        else:
            prefix = "Your diagnosis may be inaccurate"
        prompt = prefix + "The second picture is the matching result and probability " + context_str_l + " Please compare the two pictures and check the diagnosis of the first picture again." + "Give the answer in format {\"level\": "", \"reasons\": ""}"
        # prompt = self.build_diagnosis_string("", context_str_c, "", self.diagnosis_str, metadata_str, query_str)
        record_data.update({"prompt": prompt})
        return prompt, None, record_data