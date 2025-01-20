import argparse
import torch
import os, sys
import json, gc
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
sys.path.append(r"/home/hongyu/Visual-RAG-LLaVA-Med/")
from Datasets.DRDataset import DRDataset
from Datasets.eye_image_dataset import EyeImageDataset
from Datasets.MultiModalVQADataset import MultiModalVQADataset, MultiModalVQADataset2
from Datasets.lesion_balanced_dataset import LesionBalancedDataset
# from VRAG_crop import VRAG
# from VRAG_Framework.VRAG_L import VRAG
from LLavaVRAG.VRAG.VRAG_L import VRAG


class evaluation():
    def __init__(self, args, model):
        self.query_str = args.query_str
        self.model = model
        self.test_num =args.test_num
        # self.test_path = args.test_path
        self.output_path = args.output_path
        self.mode = args.mode
        if args.dataset == "DR":
            self.dataset = DRDataset(csv_file = './data/DR/multidr.csv',image_dir = './data/DR/multidr')
        if args.dataset == "ALL":
            root_path = "/home/hongyu/"
            csv_file = root_path +'partdataset/cleaned_part.csv'
            img_dir = root_path + 'partdataset/images'
            self.dataset = EyeImageDataset(csv_file=csv_file, img_dir=img_dir)
        if args.dataset == "MultiModal":
            root_path = "/home/hongyu/"
            excel_file = root_path + "Visual-RAG-LLaVA-Med/data/"+ 'Multimodal VQA Dataset/Multimodal VQA dataset_1015.xlsx'
            data_dir = root_path + "Visual-RAG-LLaVA-Med/data/" + 'Multimodal VQA Dataset'
            self.dataset = MultiModalVQADataset(excel_file, data_dir, sheet_names=args.sheet_names)
        if args.dataset == "MultiModalVQA":
            root_path = "/home/hongyu/"
            excel_file = root_path + "Visual-RAG-LLaVA-Med/data/"+ 'Multimodal VQA Dataset/Multimodal VQA dataset_1015.xlsx'
            data_dir = root_path + "Visual-RAG-LLaVA-Med/data/" + 'Multimodal VQA Dataset'
            self.dataset = MultiModalVQADataset2(excel_file, data_dir, sheet_names=args.sheet_names)
        if args.dataset == "LesionBalanced":
            root_dir = "/home/hongyu/Visual-RAG-LLaVA-Med"
            excel_file = "./data/lesion balanced dataset_new_20241210.xlsx"
            self.dataset = LesionBalancedDataset(excel_file, root_dir)
        
    def test(self):
        correct_predictions = 0
        total_samples = len(self.dataset)
        results = []
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        # Iterate through the dataloader
        idx = 0
        for images, diagnosis in tqdm(dataloader):
        # for idx in tqdm(range(total_samples)):
            if self.test_num != -1 and idx >= self.test_num:
                break
            idx += 1
            diagnosis = diagnosis[0]
            img_name = images[0]
            
            # Perform inference
            if self.mode == "Normal":
                respond, record_data = self.model.inference_rag(self.query_str, img_name)
            elif self.mode == "ALL":
                respond, record_data = self.model.inference_rag_all(self.query_str, img_name)
            elif self.mode == "MulitTurn":
                respond, record_data = self.model.inference_multi_turn(self.query_str, img_name)
            elif self.mode == "MultiTurnCheck":
                respond, record_data = self.model.inference_multi_turn_check(self.query_str, img_name)
                
            
            # Check if diagnosis is in respond
            is_correct = diagnosis in respond
            if diagnosis == "proliferative diabetic retinopathy" and "nonproliferative diabetic retinopath" in respond:
                is_correct = False
            if diagnosis == "proliferative diabetic retinopathy" and "severe nonproliferative diabetic retinopathy" in respond:
                is_correct = False
            if diagnosis == "PDR" and "NPDR" in respond:
                is_correct = False
            if is_correct:
                correct_predictions += 1

            # Append results to the list
            results.append({
                'img_name': img_name,
                'ground truth': diagnosis,
                'llm respond': respond,
                'record_data': record_data,
                'correct': is_correct
            })
            # Calculate accuracy
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            if idx % 5 == 0:
                self.save_results(results, accuracy)
                results = []
            # gc.collect()
            # torch.cuda.empty_cache()


        # Save results to a JSON file
        # with open(self.output_path, 'w') as json_file:
        #     json.dump({'accuracy': accuracy, 'results': results}, json_file, indent=4)
        self.save_results(results, accuracy)

        return accuracy
    
    def save_results(self, results, accuracy):
        # Load existing data from file if it exists
        try:
            with open(self.output_path, 'r') as json_file:
                existing_data = json.load(json_file)
                existing_results = existing_data.get('results', [])
        except FileNotFoundError:
            existing_results = []

        # Append new results to existing ones
        updated_results = existing_results + results

        # Save the updated data back to the JSON file without calculating accuracy
        with open(self.output_path, 'w') as json_file:
            json.dump({'results': updated_results}, json_file, indent=4)

        # If we need to calculate and save accuracy, do it here
        # Save the final accuracy
        with open(self.output_path, 'w') as json_file:
            json.dump({'accuracy': accuracy, 'results': updated_results}, json_file, indent=4)
    def test2(self):
        total_samples = len(self.dataset)
        results = []
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        idx = 0
        for images, diagnosis, query, answer in tqdm(dataloader):
            if self.test_num != -1 and idx >= self.test_num:
                break
            idx += 1
            diagnosis = diagnosis[0]
            img_name = images[0]
            if self.mode == "ALL":
                respond, record_data = self.model.inference_rag_all(query[0], img_name)
            results.append({
                'img_name': img_name,
                'diagnosis': diagnosis,
                'llm respond': respond,
                'record_data': record_data,
                'query': query[0],
                'answer': answer[0]
            })
        df = pd.DataFrame(results)

        # 将 DataFrame 保存为 CSV 文件
        df.to_csv(self.output_path, index=False, encoding='utf-8')
        
    def test_lesion_balanced(self):
        results = []
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        idx = 0
        for images, query, answer in tqdm(dataloader):
            if self.test_num != -1 and idx >= self.test_num:
                break
            idx += 1
            img_name = images[0]
            if self.mode == "ALL":
                respond, record_data = self.model.inference_rag_all(query[0], img_name)
            results.append({
                'img_name': img_name,
                'llm respond': respond,
                'record_data': record_data,
                'query': query[0],
                'answer': answer[0]
            })
            df = pd.DataFrame(results)

        # 将 DataFrame 保存为 CSV 文件
        df.to_csv(self.output_path, index=False, encoding='utf-8')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k-c", type=int, default=3)
    parser.add_argument("--top-k-l", type=int, default=1)
    parser.add_argument("--top-k-cl", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--meta-data", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/data/segmentation.json")
    parser.add_argument("--dataset", type=str, default="DR")
    parser.add_argument("--query-str", type=str, default="what's the diagnosis?")
    parser.add_argument("--use-pics", type=bool, default=False)
    parser.add_argument("--use-rag", type=bool, default=False)
    parser.add_argument("--test-num", type=int, default=-1)
    parser.add_argument("--image-folder", type=str, default="./segmentation/")
    parser.add_argument("--output-path", type=str, default="./output/DR.json")
    parser.add_argument("--chunk-m", type=int, default=1)
    parser.add_argument("--chunk-n", type=int, default=1)
    parser.add_argument("--tmp-path", type=str, default="./data/tmp")
    parser.add_argument("--crop-emb-path", type=str, default=None)
    parser.add_argument("--level-emb-path", type=str, default=None)
    parser.add_argument("--classic-emb-path", type=str, default=None)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--mode", type=str, default="Normal")
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--t-check', type=float, default=0.7)
    parser.add_argument('--t-filter', type=float, default=0.5)
    parser.add_argument("--sheet-names", nargs='+', type=str, default=["CFP"])
    args = parser.parse_args()
    vrag = VRAG(args) # llava, llava-med, llava-med-rag
    # vrag = InternVL2(args)
    eva = evaluation(args, vrag)
    if args.dataset == "MultiModalVQA":
        eva.test2()
    elif args.dataset == "LesionBalanced":
        eva.test_lesion_balanced()
    elif args.dataset == "MultiModal":
        eva.test()