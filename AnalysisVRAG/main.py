import os,sys
sys.path.append("/home/hongyu/Visual-RAG-LLaVA-Med")
import json
import argparse
from AnalysisVRAG.mulit_modal_vqa_analysis import MultiVQAAnalysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default="")
    parser.add_argument("--res-path", type=str, default="")
    parser.add_argument("--level-emb", type=bool, default=False)
    parser.add_argument("--num", type=int, default=-1)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--sheet-names", type=str, nargs='+', default=["CFP"])
    args = parser.parse_args()
    analysis = MultiVQAAnalysis(args)
    cm = analysis.calculate_confusion_matrix(args.res_path)
    print(f"Accuracy: {analysis.calculate_accuracy():.4f}")
    print(cm)