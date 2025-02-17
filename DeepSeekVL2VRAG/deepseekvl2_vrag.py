import sys, pathlib

sys.path.append("/home/hongyu/Visual-RAG-LLaVA-Med")

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from DeepSeekVL2VRAG.DeepSeek_VL2.deepseek_vl2.models import (
    DeepseekVLV2Processor,
    DeepseekVLV2ForCausalLM,
)
from DeepSeekVL2VRAG.DeepSeek_VL2.deepseek_vl2.utils.io import load_pil_images

from PIL import Image

def resize_images(images: list[Image.Image], size: tuple) -> list[Image.Image]:
    """
    调整给定图像列表的分辨率。

    Args:
        images (List[PIL.Image.Image]): 需要调整大小的PIL图像列表。
        size (tuple): 一个包含宽度和高度的目标尺寸元组，例如(800, 600)。

    Returns:
        resized_images (List[PIL.Image.Image]): 调整过分辨率的图像列表。
    """

    resized_images = []

    for img in images:
        # 使用ANTIALIAS算法来保证缩放质量
        resized_img = img.resize(size, Image.Resampling.LANCZOS)
        resized_images.append(resized_img)

    return resized_images


class DeepSeekVL2VRAG:
    def __init__(
        self,
        checkpoint_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/deepseek-vl2-small",
    ):

        # specify the path to the model
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.vl_chat_processor: DeepseekVLV2Processor = (
            DeepseekVLV2Processor.from_pretrained(checkpoint_path, min_pixels=min_pixels, max_pixels=max_pixels)
        )
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    def inference(
        self,
        conversation: list,
    ):

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        pil_images = resize_images(pil_images, (768, 768))
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        ).to(self.vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=False
        )
        # print(f"{prepare_inputs['sft_format'][0]}", answer)
        return answer

    def inference_rag(
        self,
        image_index_folder: pathlib.Path = None,
        text_emb_folder: pathlib.Path = None,
        use_pics: int = 0
    ):
        pass

if __name__ == "__main__":
    image_path = "./data/Classic Images/DR/mild NPDR_1.jpeg"
    query = "What is the diagnsis"
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n {query}.",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    dsvl2 = DeepSeekVL2VRAG()
    answer = dsvl2.inference(conversation=conversation)
    print(answer)
