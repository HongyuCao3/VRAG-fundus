import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd()))
import torch
from PIL import Image
import argparse
from dataclasses import dataclass
from transformers import set_seed
from LLavaVRAG.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from LLavaVRAG.llava.conversation import conv_templates, SeparatorStyle
from LLavaVRAG.llava.model.builder import load_pretrained_model
from LLavaVRAG.llava.utils import disable_torch_init
from LLavaVRAG.llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
    process_images,
)
from fundus_knowledge_base.index_manager.mulit_disease_index_manager import (
    MultiDiseaseIndexManager,
)
from ContextFormer.ClassificationContextFormer import (
    ClassificationContextFormer,
    ClassificationContextConfig,
)
from PathManager.EmbPathManager import EmbPathManager, EmbPathConfig
from fundus_knowledge_base.knowledge_retriever.TextRetriever import TextRetriever
from conflict_resolution.vrag_filter import VRAGFilter
from conflict_resolution.checker import Checker


@dataclass
class LLaVAConfig:
    llava_med_path = "/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b"
    llava_med_finetuned_path = "/home/hongyu/eye_llava_medllava_finetune_mistral"


@dataclass
class LLaVAInferenceParams:
    filter: bool = False  # 是否过滤
    check: bool = False  # 是否检查
    num_beams: int = 1  # 使用的beam数量
    temperature: float = 0.0  # 温度参数，用于控制生成的随机性
    image_index_folder: pathlib.Path = None  # 图像索引文件夹路径
    image_topk: int = 1  # 返回图像结果的数量
    text_emb_folder: pathlib.Path = None  # 文本嵌入文件夹路径
    text_topk: int = 1  # 返回文本结果的数量
    use_pics: int = 0  # 是否使用图片，以及使用的图片数量


class llava_vrag:
    def __init__(
        self,
        args,
        model_path,
        model_base=None,
        sheet_names: list[str] = ["CFP"],
        t_filter: float = 0.5,
        t_check: float = 0.5,
    ):
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(model_path, model_base, model_name)
        )
        self.conv_mode = args.conv_mode
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_beams = args.num_beams
        self.context_former = ClassificationContextFormer()
        self.vrag_filter = VRAGFilter(
            self.context_former, image_threshold=t_filter, sheet_names=sheet_names
        )
        self.checker = Checker(threshold=t_check)

    def model_chat(self, images, prompt):
        set_seed(0)
        disable_torch_init()
        qs = prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        if self.model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        image_tensor = process_images(images, self.image_processor, self.model.config)[
            0
        ]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        return outputs

    def inference(
        self, query: str, image_path: pathlib.Path, params: LLaVAInferenceParams
    ):
        if params.image_index_folder:
            self.index_manager = MultiDiseaseIndexManager()
            self.image_index = self.index_manager.load_index(params.image_index_folder)
        if params.text_emb_folder:
            self.text_embedding = TextRetriever(emb_folder=params.text_emb_folder)

        # retrieval and post-process
        if params.image_index_folder:
            retrieved_images = self.index_manager.retrieve_image(
                self.image_index, img_path=image_path, top_k=params.image_topk
            )
            if params.filter:
                retrieved_images = self.vrag_filter.filter_retrieved_images(
                    retrieved_images=retrieved_images
                )
            image_context = " ".join(
                [
                    f"{txt}: {score}"
                    for txt, score in zip(retrieved_images.txt, retrieved_images.score)
                ]
            )
        else:
            image_context = None
        if params.text_emb_folder:
            retrieved_texts = self.text_embedding.retrieve(
                input_img=image_path, k=params.text_topk
            )
            if params.filter:
                retrieved_texts = self.vrag_filter.filter_retrieved_texts(
                    retrieved_texts=retrieved_texts
                )
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

        image_org = Image.open(image_path)
        images = [image_org]
        for i in range(params.use_pics):
            images.append(retrieved_images.img[i])

        outputs = self.model_chat(images, prompt)
        record_data["outputs"] = outputs
        return outputs, record_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--image-folder", type=str, default="./segmentation/")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--meta-data",
        type=str,
        default="/home/hongyu/Visual-RAG-LLaVA-Med/data/segmentation.json",
    )
    parser.add_argument("--chunk-m", type=int, default=1)
    parser.add_argument("--chunk-n", type=int, default=1)
    parser.add_argument("--tmp-path", type=str, default="./data/tmp")
    args = parser.parse_args()
    config = LLaVAConfig()
    lv = llava_vrag(
        args=args,
        model_path=config.llava_med_path,
    )
    test_img = pathlib.Path("/home/hongyu/DDR/lesion_segmentation/test/image/007-1789-100.jpg")
    query_str_0 = "Can you describe the image in details?"
    query_str_1 = "what's the diagnosis?"
    params = LLaVAInferenceParams()
    output, records = lv.inference(query=query_str_1, image_path=test_img, params=params)
    print(output)
    print(records)
