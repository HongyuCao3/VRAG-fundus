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


class QwenVLVRAG:
    def __init__(self, checkpoint_path="./Model/Qwen2.5-VL-7B-Instruct"):
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.model =  Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        )
        self.processor = AutoProcessor.from_pretrained(checkpoint_path, min_pixels=min_pixels, max_pixels=max_pixels)

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
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
    
    
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
    answer = qvlvrag.inference(messages=messages)
    print(answer)