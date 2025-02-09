import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd()))
import torch
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
        model_base,
        sheet_names: str,
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
        
        # TODO: do inference
