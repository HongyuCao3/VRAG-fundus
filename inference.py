import argparse
import torch
import os
import json
from tqdm import tqdm

from PIL import Image
import math
from transformers import set_seed, logging

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

def inference(args):
    set_seed(0)
    disable_torch_init()
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    with open(args.question_file, 'r', encoding='utf-8') as file:
        questions = json.load(file)
    # questions = [json.loads(q) for q in open(args.question_file,"r")]
    for line in tqdm(questions):
        qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        image_file = line["image"]
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
        # TODO:批量测试需要记录上下文信息
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/data/eye_diag.json")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--image-folder", type=str, default="./segmentation/")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    
    print(inference(args))