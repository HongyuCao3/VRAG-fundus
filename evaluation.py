import argparse
import torch
import os
import json
from tqdm import tqdm
from dataset import DRDataset
from torch.utils.data import Dataset, DataLoader

from VRAG_crop import VRAG


class evaluation():
    def __init__(self, args, model):
        self.query_str = args.query_str
        self.model = model
        # self.test_path = args.test_path
        self.output_path = args.output_path
        if args.dataset == "DR":
            self.dataset = DRDataset(csv_file = './data/DR/multidr.csv',image_dir = './data/DR/multidr')
        self.test_num = args.test_num
        
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
            respond, record_data = self.model.inference_rag(self.query_str, img_name)
            
            # Check if diagnosis is in respond
            is_correct = diagnosis in respond
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

        # Save results to a JSON file
        with open(self.output_path, 'w') as json_file:
            json.dump({'accuracy': accuracy, 'results': results}, json_file, indent=4)


        return accuracy
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--meta-data", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/data/segmentation.json")
    parser.add_argument("--dataset", type=str, default="DR")
    parser.add_argument("--query-str", type=str, default="what's the diagnosis accroding to the diagnosising standard?")
    parser.add_argument("--use-pics", type=bool, default=False)
    parser.add_argument("--use-rag", type=bool, default=False)
    parser.add_argument("--test-num", type=int, default=-1)
    parser.add_argument("--image-folder", type=str, default="./segmentation/")
    parser.add_argument("--output-path", type=str, default="./output/DR.json")
    parser.add_argument("--chunk-m", type=int, default=1)
    parser.add_argument("--chunk-n", type=int, default=1)
    parser.add_argument("--tmp-path", type=str, default="./data/tmp")
    parser.add_argument("--crop-emb-path", type=str, default="./data/emb_crop")
    parser.add_argument("--level-emb-path", type=str, default="./data/level_emb")
    args = parser.parse_args()
    vrag = VRAG(args) # llava, llava-med, llava-med-rag
    eva = evaluation(args, vrag)
    eva.test()