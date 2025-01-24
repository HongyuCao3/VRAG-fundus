import sys
sys.path.append(r"/home/hongyu/Visual-RAG-LLaVA-Med/")
import pathlib
import torch
from KnowledgeBase.Retriever.ImageRetriever import ImageRetriever
from KnowledgeBase.Retriever.TextRetriever import TextRetriever


class InternVL2_finetuned:
    def __init__(self):
        self.image_retriever = ImageRetriever()
        self.text_retriever = TextRetriever()
        
    def inference_rag(self, query: str, img_path: pathlib.Path, filter: bool = False):
        # do retireval
        similar_imgs = self.image_retriever.get_similar_images(img_path)
        similar_txts = self.text_retriever.get_simliar_texts(img_path)

        # filter finetuned
        if filter:
            similar_imgs = self.filter.filter_finetuned(similar_imgs)
            similar_txts = self.filter.filter_finetuned(similar_txts)

        # form inference context
        prompt, images, record_data = self.context_former.form_inference_rag_context(
            img_path, query, similar_imgs, similar_txts
        )

        # do inference
        generation_config = dict(
            num_beams=self.num_beams,
            # max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
            min_new_tokens=1,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
        )
        pixel_values = self.load_image(img_path, max_num=12).to(torch.float16).cuda()
        response = self.model.chat(
            self.tokenizer, pixel_values, prompt, self.generation_config, verbose=True
        )
        return response, record_data
