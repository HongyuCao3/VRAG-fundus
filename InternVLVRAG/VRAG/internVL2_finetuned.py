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
from InternVLVRAG.VRAG.internVL2_base import InternVL2Base, InterVLInferenceParams


class InternVL2Finetuned(InternVL2Base):
    def __init__(
        self,
        args: argparse.Namespace,
        sheet_names: str,
        t_filter: float=0.5,
        t_check: float=0.5,
    ):
        self.model, self.tokenizer = load_model_and_tokenizer(args)
        self.context_former = ClassificationContextFormer()
        self.vrag_filter = VRAGFilter(
            self.context_former, image_threshold=t_filter, sheet_names=sheet_names
        )
        self.checker = Checker(threshold=t_check)

    def inference(
        self,
        query: str,
        image_path: pathlib.Path,
        params: InterVLInferenceParams
    ):
        if params.image_index_folder:
            self.index_manager = MultiDiseaseIndexManager()
            self.image_index = self.index_manager.load_index(params.image_index_folder)
        if text_emb_folder:
            self.text_embedding = TextRetriever(emb_folder=params.text_emb_folder)

        # retrieval and post-process
        if image_index_folder:
            retrieved_images = self.index_manager.retrieve_image(
                self.image_index, img_path=image_path, top_k=params.image_topk
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
            retrieved_texts = self.text_embedding.retrieve(input_img=image_path, k=params.text_topk)
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
            num_beams=params.num_beams,
            min_new_tokens=1,
            do_sample=True if params.temperature > 0 else False,
            temperature=params.temperature,
        )
        pixel_values = self.load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        for i in range(params.use_pics):
            rag_pixel_values = self.load_image(retrieved_images.img[i], max_num=12).to(torch.bfloat16).cuda()
            pixel_values = torch.cat((pixel_values, rag_pixel_values), dim=0)
        response = self.model.chat(
            self.tokenizer, pixel_values, prompt, generation_config, verbose=True
        )
        return response, record_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B")
    parser.add_argument("--dataset", type=str, default="DR")
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()
    query = "what's the diagnosis level?"
    test_img = "/home/hongyu/DDR/lesion_segmentation/test/image/007-1789-100.jpg"
    pm = EmbPathManager()
    text_emb_folder = pm.get_emb_dir(pm.config.default_text_emb_name)
    image_index_folder = pathlib.Path(
        "./fundus_knowledge_base/emb_savings/mulit_desease_image_index"
    )
    params = InterVLInferenceParams(filter=False, check=False, image_index_folder=image_index_folder, text_emb_folder=text_emb_folder)
    I2F = InternVL2Finetuned(args = args, sheet_names=["CFP"])
    answer = I2F.inference(query=query, image_path=test_img, params=params)
    print(answer)
    