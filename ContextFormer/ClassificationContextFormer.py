import sys
from torch.types import (
    Number,
)

sys.path.append("/home/hongyu/Visual-RAG-LLaVA-Med")
from PIL import Image
from llama_index.core.schema import ImageDocument
import pathlib
from ContextFormer.BaseContextFormer import BaseContextFormer, BaseContextConfig


class ClassificationContextConfig(BaseContextConfig):
    DR_diagnosing_level = {
        "Normal": "No lesion",
        "Mild NPDR": "MAs only",
        "Moderate NPDR": "At least one hemorrhage or MA and/or at least one of the following: Retinal hemorrhages, Hard exudates, Cotton wool spots, Venous beading",
        "Severe NPDR": "Any of the following but no signs of PDR (4-2-1 rule): >20 intraretinal hemorrhages in each of four quadrants, definite venous, beading in two or more quadrants, Prominent IRMA in one or more quadrants",
        "PDR": "One of either: Neovascularization, Vitreous/preretinal hemorrhage",
    }
    DR_level_mapping = {
        "Normal": "Normal",
        "Mild NPDR": "mild nonproliferative diabetic retinopathy",
        "Moderate NPDR": "moderate nonproliferative diabetic retinopathy",
        "Severe NPDR": "severe nonproliferative diabetic retinopathy",
        "PDR": "proliferative diabetic retinopathy",
    }
    lesion = {
        "microaneurysm": "",
        "hemorrhage": "",
        "cotton wool spots": "",
        "exudates": "",
    }
    # TODO:添加用于多病种分类的dict


class ClassificationContextFormer(BaseContextFormer):
    def __init__(self):
        super().__init__()
        self.config = ClassificationContextConfig()

    def form_context_all_cl(
        self, img_path: str, query: str, diagnosis_context: dict, use_pics: bool = False
    ):
        record_data = {}
        record_data.update({"ret_cl": str(diagnosis_context)})
        record_data.update({"org": img_path})
        image_documents = [ImageDocument(image_path=img_path)]
        image_org = Image.open(img_path)
        images = [image_org]
        img = diagnosis_context["img"]
        if use_pics:
            for res_img in img:
                image_documents.append(ImageDocument(image_path=res_img))
                # print(res_img)
                image = Image.open(res_img)
                images.append(image)

        result_dict_cl = dict(zip(diagnosis_context["txt"], diagnosis_context["score"]))
        diagnosis_context_str = str(result_dict_cl)
        metadata_str = diagnosis_context["metadata"]
        metadata_str.extend(diagnosis_context["metadata"])
        prompt = self.build_prompt(diagnosis_context=diagnosis_context_str, query=query)
        record_data.update({"prompt": prompt})
        return prompt, images, record_data

    def build_prompt(
        self,
        query: str,
        image_context: str = None,
        text_context: str = None,
        diagnosis_standard: str = None,
    ):
        parts = []
        if diagnosis_standard:
            parts.append(f"Diagnosing Standard: {diagnosis_standard}\n")
        if image_context:
            parts.append(
                f"The possible diagnosing level and similarity: {image_context}\n"
            )
        if text_context:
            parts.append(f"The possible diagnosis and similarity: {text_context}\n")
        parts.append(query)

        return "".join(parts)

    def build_query_context(
        self,
        image_path: pathlib.Path,
        query: str,
        image_context: str = None,
        text_context: str = None,
        diagnosis_standard: str = None,
    ):
        record_data = {}
        record_data["similar_imgs"] = image_context
        record_data["similar_txts"] = text_context
        prompt = self.build_prompt(
            query=query,
            image_context=image_context,
            text_context=text_context,
            diagnosis_standard=diagnosis_standard,
        )
        return prompt, record_data
