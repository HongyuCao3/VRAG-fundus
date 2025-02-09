import sys

sys.path.append(str(pathlib.Path.cwd()))
import pathlib
import torch
import argparse
from PIL import Image
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
        self.vrag_filter = VRAGFilter(
            self.context_former, threshold=t_filter, sheet_names=sheet_names
        )
        self.checker = Checker(threshold=t_check)

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert("RGB")
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def inference_rag(
        self,
        query: str,
        image_path: pathlib.Path,
        filter: bool = False,
        check: bool = False,
        input_pics_num: int = 0,
        num_beams: int = 1,
        temperature: float = 0,
        image_index_folder: pathlib.Path = None,
        text_emb_folder: pathlib.Path = None,
        use_pics: int = 0,
    ):
        if image_index_folder:
            self.index_manager = MultiDiseaseIndexManager()
            self.image_index = self.index_manager.load_index(image_index_folder)
        if text_emb_folder:
            self.text_embedding = TextRetriever(emb_folder=text_emb_folder)

        # form inference context
        if image_index_folder:
            retrieved_images = self.index_manager.retrieve_image(
                self.image_index, img_path=image_path, top_k=1
            )
        if text_emb_folder:
            retrieved_texts = self.text_embedding.retrieve(input_img=image_path)
        if image_index_folder:
            image_context = " ".join(
                [
                    f"{txt}: {score}"
                    for txt, score in zip(retrieved_images.txt, retrieved_images.score)
                ]
            )
        else:
            image_context = None
        if text_emb_folder:
            text_context = " ".join(
                [
                    f"{txt}: {score}"
                    for txt, score in zip(retrieved_texts.txt, retrieved_texts.score)
                ]
            )
        else:
            text_context = None
        prompt, images, record_data = self.context_former.form_rag_context(
            image_path, query, image_context, text_context
        )

        # do inference
        generation_config = dict(
            num_beams=num_beams,
            # max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
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
        # TODO: add filter
        return response, record_data
