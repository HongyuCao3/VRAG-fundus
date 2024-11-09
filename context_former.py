import argparse
import torch
import os
import json
from tqdm import tqdm

from PIL import Image
from llama_index.core.schema import ImageDocument

class ContextFormer():
    def __init__(self, use_pics):
        self.use_pics = use_pics
        self.diagnosing_level = {"Normal": "No lesion","Mild NPDR": "MAs only", "Moderate NPDR": "At least one hemorrhage or MA and/or at least one of the following: Retinal hemorrhages, Hard exudates, Cotton wool spots, Venous beading", "Severe NPDR": "Any of the following but no signs of PDR (4-2-1 rule): >20 intraretinal hemorrhages in each of four quadrants, definite venous, beading in two or more quadrants, Prominent IRMA in one or more quadrants", "PDR": "One of either: Neovascularization, Vitreous/preretinal hemorrhage"}
        self.diagnosis_str = ""
        for key, value in self.diagnosing_level.items():
            self.diagnosis_str += f"{key}: {value}"
        self.ret_empty = {"img": [], "txt": [], "score": [], "metadata": []}
    
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