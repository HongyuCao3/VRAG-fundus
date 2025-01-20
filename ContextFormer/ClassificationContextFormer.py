import sys

sys.path.append("/home/hongyu/Visual-RAG-LLaVA-Med")
from PIL import Image
from llama_index.core.schema import ImageDocument
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


class ClassificationContextFormer(BaseContextFormer):
    def __init__(self):
        super().__init__()
        self.config = ClassificationContextConfig()

    def build_query(
        self,
        query: str,
        level_context: str = None,
        crop_lesion_context: str = None,
        diagnosis_context: str = None,
        diagnosis_standard: str = None,
    ):
        parts = []
        if diagnosis_standard:
            parts.append(f"Diagnosing Standard: {diagnosis_standard}\n")
        if level_context:
            parts.append(
                f"The possible diagnosing level and similarity: {level_context}\n"
            )
        if crop_lesion_context:
            parts.append(f"The possible lesion and similarity: {crop_lesion_context}\n")
        if diagnosis_context:
            parts.append(
                f"The possible diagnosis and similarity: {diagnosis_context}\n"
            )
        parts.append(query)

        return "".join(parts)

    def form_context_all_cl(
        img_path: str, query: str, ret_cl: dict, use_pics: bool = False
    ):
        record_data = {}
        record_data.update({"ret_cl": str(ret_cl)})
        record_data.update({"org": img_path})
        image_documents = [ImageDocument(image_path=img_path)]
        image_org = Image.open(img_path)
        images = [image_org]
        img = ret_cl["img"]
        if use_pics:
            for res_img in img:
                image_documents.append(ImageDocument(image_path=res_img))
                # print(res_img)
                image = Image.open(res_img)
                images.append(image)

        result_dict_cl = dict(zip(ret_cl["txt"], ret_cl["score"]))
        context_str_cl = str(result_dict_cl)
        metadata_str = ret_cl["metadata"]
        metadata_str.extend(ret_cl["metadata"])
        prompt = self.build_diagnosis_string_all(
            "", "", context_str_cl, "", metadata_str, query_str
        )
        record_data.update({"prompt": prompt})
        return prompt, images, record_data
