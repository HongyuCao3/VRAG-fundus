import argparse
import torch
import os
import json
from tqdm import tqdm

from VRAG_crop import VRAG


class evaluation():
    def __init__(self, args, model):
        self.query_str = args.query_str
        self.model = model
        self.test_path = args.test_path
        
    def test(self):
        for file in os.listdir(self.test_path):
            # TODO：添加test集ground truth提取代码
            respond, record_data = self.model.inference_rag(self.query_str, file)
        # TODO:添加结果保存和对比代码
    
    
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
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--meta-data", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/data/segmentation.json")
    parser.add_argument("--test-path", type=str, default="/home/hongyu/DDR/lesion_segmentation/test/image/")
    parser.add_argument("--query-str", type=str, default="what's the diagnosis?")
    parser.add_argument("--use-pics", type=bool, default=False)
    parser.add_argument("--use-rag", type=bool, default=False)
    args = parser.parse_args()
    vrag = VRAG(args)
    eva = evaluation(args, vrag)