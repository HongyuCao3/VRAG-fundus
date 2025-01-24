import sys

sys.path.append(r"/home/hongyu/Visual-RAG-LLaVA-Med/")
import pathlib
import torch
import argparse
from VRAG_Framework.vrag_filter import VRAGFilter
from VRAG_Framework.checker import Checker
from ContextFormer.ClassificationContextFormer import (
    ClassificationContextFormer,
    ClassificationContextConfig,
)
from KnowledgeBase.Retriever.ImageRetriever import ImageRetriever
from KnowledgeBase.Retriever.TextRetriever import TextRetriever
from VRAG_Framework import load_model_and_tokenizer

class InternVL2_finetuned:
    def __init__(
        self,
        args: argparse.Namespace,
        sheet_names: str,
        t_filter: float,
        t_check: float,
    ):
        self.model, self.tokenizer = load_model_and_tokenizer(args) 
        self.context_former = ClassificationContextFormer()
        self.image_retriever = ImageRetriever()
        self.text_retriever = TextRetriever()
        self.vrag_filter = VRAGFilter(
            self.context_former, threshold=t_filter, sheet_names=sheet_names
        )
        self.checker = Checker(threshold=t_check)

    def inference_rag(
        self,
        query: str,
        img_path: pathlib.Path,
        filter: bool = False,
        check: bool = False,
        input_pics_num: int = 0,
        num_beams: int=1,
        temperature: float=0,
    ):
        # do retireval
        similar_imgs = self.image_retriever.get_similar_images(img_path)
        similar_txts = self.text_retriever.get_similar_texts(img_path)

        # filter finetuned
        if filter:
            similar_imgs = self.vrag_filter.filter_finetuned(similar_imgs)
            similar_txts = self.vrag_filter.filter_finetuned(similar_txts)

        # form inference context
        prompt, images, record_data = self.context_former.form_rag_context(
            img_path, query, similar_imgs, similar_txts
        )

        # do inference
        generation_config = dict(
            num_beams=num_beams,
            # max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
            min_new_tokens=1,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
        )
        # TODO: add images number 
        pixel_values = self.load_image(img_path, max_num=12).to(torch.float16).cuda()
        response = self.model.chat(
            self.tokenizer, pixel_values, prompt, generation_config, verbose=True
        )
        # TODO: add check
        return response, record_data
