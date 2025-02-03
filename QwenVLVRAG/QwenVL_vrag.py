from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd()))
from Datasets.MultiModalClassificationDataset import (
    MultiModalClassificationDataset,
    MultiModalClassificationConfig,
)
from qwen_vl_utils import process_vision_info
from fundus_knowledge_base.index_manager.mulit_disease_index_manager import (
    MultiDiseaseIndexManager,
)
from PathManager.EmbPathManager import EmbPathManager, EmbPathConfig
from fundus_knowledge_base.knowledge_retriever.TextRetriever import TextRetriever


class QwenVLVRAG:
    def __init__(
        self,
        checkpoint_path="./Model/Qwen2.5-VL-7B-Instruct",
    ):
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_path, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def build_prompt(
        self,
        query: str,
        image_context: str = None,
        text_context: str = None,
        diagnosis_standard: str = None,
    ):
        parts = []
        if diagnosis_standard:
            parts.append(f"Diagnosing Standard: {diagnosis_standard}\n")
        if image_context:
            parts.append(
                f"The possible diagnosing level and similarity: {image_context}\n"
            )
        if text_context:
            parts.append(f"The possible diagnosis and similarity: {text_context}\n")
        parts.append(query)

        return "".join(parts)

    def inference(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text

    def inference_rag(
        self,
        messages,
        image_index_folder: pathlib.Path = None,
        text_emb_folder: pathlib.Path = None,
        use_pics: bool = False,
    ):
        if image_index_folder:
            self.index_manager = MultiDiseaseIndexManager()
            self.image_index = self.index_manager.load_index(image_index_folder)
        if text_emb_folder:
            self.text_embedding = TextRetriever(emb_folder=text_emb_folder)
        messages_rag = []
        for message in messages:
            image_path = message["content"][0]["image"]
            query = message["content"][1]["text"]
            if image_index_folder:
                retrieved_images = self.index_manager.retrieve_image(
                    self.image_index, img_path=image_path, top_k=1
                )
            if text_emb_folder:
                retrieved_texts = self.text_embedding.retrieve(input_img=image_path)
            content = [message["content"][0]]
            if use_pics and image_index_folder:  # multi image as input
                for img in retrieved_images["img"]:
                    content.append({"type": "image", "image": img})
            # form prompt
            if image_index_folder:
                image_context = " ".join(
                    [
                        f"{txt}: {img}"
                        for txt, img in zip(
                            retrieved_images["txt"], retrieved_images["score"]
                        )
                    ]
                )
            else:
                image_context = None
            if text_emb_folder:
                text_context = " ".join(
                    [
                        f"{txt}: {img}"
                        for txt, img in zip(
                            retrieved_texts["txt"], retrieved_texts["score"]
                        )
                    ]
                )
            else:
                text_context = None
            prompt = self.build_prompt(
                query=query, image_context=image_context, text_context=text_context
            )
            content.append({"type": "text", "text": prompt})
            messages_rag.append({"role": "user", "content": content})
        answer = self.inference(messages_rag)
        return answer


if __name__ == "__main__":
    qvlvrag = QwenVLVRAG()
    image_path = "./data/Classic Images/DR/mild NPDR_1.jpeg"
    query = "What is the diagnsis"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": query},
            ],
        }
    ]
    pm = EmbPathManager()
    text_emb_folder = pm.get_emb_dir(pm.config.default_text_emb_name)
    image_index_folder = pathlib.Path(
        "./fundus_knowledge_base/emb_savings/mulit_desease_image_index"
    )
    answer = qvlvrag.inference_rag(
        messages=messages,
        text_emb_folder=text_emb_folder,
        image_index_folder=image_index_folder,
    )
    print(answer)
