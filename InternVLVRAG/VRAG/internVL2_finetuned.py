import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd()))
import torch
import argparse
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from conflict_resolution.vrag_filter import VRAGFilter
from conflict_resolution.checker import Checker
from ContextFormer.ClassificationContextFormer import (
    ClassificationContextFormer,
    ClassificationContextConfig,
)
from fundus_knowledge_base.index_manager.mulit_disease_index_manager import (
    MultiDiseaseIndexManager,
)
from PathManager.EmbPathManager import EmbPathManager, EmbPathConfig
from fundus_knowledge_base.knowledge_retriever.TextRetriever import TextRetriever
from InternVLVRAG.internvl.model import load_model_and_tokenizer
from InternVLVRAG.VRAG.internVL2_base import InternVL2Base


class InternVL2Finetuned(InternVL2Base):
    def __init__(
        self,
        args: argparse.Namespace,
        sheet_names: str,
        t_filter: float,
        t_check: float,
    ):
        self.model, self.tokenizer = load_model_and_tokenizer(args)
        self.context_former = ClassificationContextFormer()
        self.vrag_filter = VRAGFilter(
            self.context_former, threshold=t_filter, sheet_names=sheet_names
        )
        self.checker = Checker(threshold=t_check)

    def inference_rag(
        self,
        query: str,
        image_path: pathlib.Path,
        filter: bool = False,
        check: bool = False,
        num_beams: int = 1,
        temperature: float = 0,
        image_index_folder: pathlib.Path = None,
        image_topk:int=1,
        text_topk: int=1,
        text_emb_folder: pathlib.Path = None,
        use_pics: int = 0,
        # TODO: warp the parameters
    ):
        if image_index_folder:
            self.index_manager = MultiDiseaseIndexManager()
            self.image_index = self.index_manager.load_index(image_index_folder)
        if text_emb_folder:
            self.text_embedding = TextRetriever(emb_folder=text_emb_folder)

        # retrieval and post-process
        if image_index_folder:
            retrieved_images = self.index_manager.retrieve_image(
                self.image_index, img_path=image_path, top_k=image_topk
            )
            if filter:
                retrieved_images = self.vrag_filter.filter_retrieved_images(retrieved_images=retrieved_images)
            image_context = " ".join(
                [
                    f"{txt}: {score}"
                    for txt, score in zip(retrieved_images.txt, retrieved_images.score)
                ]
            )
        else:
            image_context = None
        if text_emb_folder:
            retrieved_texts = self.text_embedding.retrieve(input_img=image_path, k=text_topk)
            if filter:
                retrieved_texts = self.vrag_filter.filter_retrieved_texts(retrieved_texts=retrieved_texts)
            text_context = " ".join(
                [
                    f"{txt}: {score}"
                    for txt, score in zip(retrieved_texts.txt, retrieved_texts.score)
                ]
            )
        else:
            text_context = None
        prompt, record_data = self.context_former.build_query_context(
            image_path=image_path,
            query=query,
            image_context=image_context,
            text_context=text_context,
        )

        # do inference
        generation_config = dict(
            num_beams=num_beams,
            min_new_tokens=1,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
        )
        pixel_values = self.load_image(image_path, max_num=12).to(torch.float16).cuda()
        for i in range(use_pics):
            rag_pixel_values = self.load_image(retrieved_images["img"][i])
            pixel_values = torch.cat((pixel_values, rag_pixel_values), dim=0)
        response = self.model.chat(
            self.tokenizer, pixel_values, prompt, generation_config, verbose=True
        )
        return response, record_data
