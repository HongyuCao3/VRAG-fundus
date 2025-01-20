import sys

sys.path.append("/home/hongyu/Visual-RAG-LLaVA-Med")
from PIL import Image
from llama_index.core.schema import ImageDocument
from ContextFormer.BaseContextFormer import BaseContextFormer, BaseContextConfig

class VQAContextConfig(BaseContextConfig):
    pass

class VQAContextFormer(BaseContextFormer):
    def __init__(self):
        super().__init__()
    
    def form_context_all_cl(self, img_path: str, query: str, diagnosis_context: dict):
        record_data = {}
        record_data.update({"ret_cl": str(diagnosis_context)})
        record_data.update({"org": img_path})
        image_documents = [ImageDocument(image_path=img_path)]
        image_org = Image.open(img_path)
        images= [image_org]
        img = diagnosis_context["img"]
        if self.use_pics:
            for res_img in img:
                image_documents.append(ImageDocument(image_path=res_img))
                # print(res_img)
                image = Image.open(res_img)
                images.append(image)
        result_dict_cl = dict(zip(diagnosis_context["txt"], diagnosis_context["score"]))
        diagnosis_context_str = str(result_dict_cl)
        metadata_str = diagnosis_context["metadata"]
        metadata_str.extend(diagnosis_context["metadata"])
        prompt = self.build_prompt(
            query=query,
            diagnosis_context=diagnosis_context_str
        )
        record_data.update({"prompt": prompt})
        return prompt, images, record_data