import os,sys
sys.path.append("/home/hongyu/Visual-RAG-LLaVA-Med")
import json
import argparse
from AnalysisVRAG.multi_modal_classification import MultiModalClassificationAnalysis
from AnalysisVRAG.multi_modal_vqa import MultiModalVQAAnalysisConfig, MultiModalVQAAnalysis

def analyze_multi_modal_classification():
    result_saving_path = "./QwenVLVRAG/output/Multimodal VQA Dataset/None_None_usepics_False_-1.json"
    image_saving_path = "./tmp.png"
    mmca = MultiModalClassificationAnalysis(file_path=result_saving_path, sheet_names=["CFP"])
    mmca.calculate_confusion_matrix(image_saving_path=image_saving_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default="")
    parser.add_argument("--res-path", type=str, default="")
    parser.add_argument("--level-emb", type=bool, default=False)
    parser.add_argument("--num", type=int, default=-1)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--sheet-names", type=str, nargs='+', default=["CFP"])
    args = parser.parse_args()
    # analysis = MultiModalClassificationAnalysis(args)
    # cm = analysis.calculate_confusion_matrix(args.res_path)
    # print(f"Accuracy: {analysis.calculate_accuracy():.4f}")
    # print(cm)
    analyze_multi_modal_classification()