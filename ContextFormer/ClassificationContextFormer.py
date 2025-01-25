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
    
    def form_rag_context(
        self,
        img_path: pathlib.Path,
        query: str,
        similar_imgs: list[tuple[pathlib.Path, Number]]=[],
        similar_txts: list[tuple[pathlib.Path, Number]]=[],
        input_pics_num: int=0
    ):
        record_data = {}
        record_data["similar_imgs"] = similar_imgs
        record_data["similar_txts"] = similar_txts
        image_documents = [ImageDocument(image_path=img_path)]
        image_org = Image.open(img_path)
        # process similar images
        images = [image_org]
        diagnosis = []
        # TODO： 查看返回类别
        for img, diag in similar_imgs:
            image_documents.append(ImageDocument(image_path=img))
            image = Image.open(img)
            images.append(image)
            diagnosis.append(diag)
            
        # process similar texts
