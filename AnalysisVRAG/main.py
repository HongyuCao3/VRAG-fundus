import sys, pathlib

sys.path.append(str(pathlib.Path.cwd()))
import json
from PathManager.EmbPathManager import EmbPathManager
import argparse
from AnalysisVRAG.multi_modal_classification import MultiModalClassificationAnalysis
from AnalysisVRAG.multi_modal_vqa import (
    MultiModalVQAAnalysisConfig,
    MultiModalVQAAnalysis,
)


def analyze_multi_modal_classification():
    pm = EmbPathManager()
    text_emb_folder = pm.get_emb_dir(pm.config.default_text_emb_name)
    image_index_folder = pathlib.Path(
        "./fundus_knowledge_base/emb_savings/mulit_desease_image_index"
    )
    # text_emb_name = text_emb_folder.name
    text_emb_name = None
    image_index_name = image_index_folder.name
    # image_index_name = None
    use_pics = False
    result_saving_path = f"./QwenVLVRAG/output/Multimodal VQA Dataset/{image_index_name}_{text_emb_name}_usepics_{use_pics}_-1.json"
    image_saving_path = result_saving_path.replace("output", "output/analysis")
    image_saving_path = image_saving_path.replace("json", "png")
    mmca = MultiModalClassificationAnalysis(
        file_path=result_saving_path, sheet_names=["CFP"]
    )
    mmca.calculate_confusion_matrix(image_saving_path=image_saving_path)


def analyze_multi_modal_vqa():
    pm = EmbPathManager()
    text_emb_folder = pm.get_emb_dir(pm.config.default_text_emb_name)
    image_index_folder = pathlib.Path(
        "./fundus_knowledge_base/emb_savings/mulit_desease_image_index"
    )
    text_emb_name = text_emb_folder.name
    image_index_name = image_index_folder.name
    result_saving_path = f"./QwenVLVRAG/output/Multimodal VQA Dataset_VQA/{image_index_name}_{text_emb_name}_usepics_0_-1.json"
    analysis_config = MultiModalVQAAnalysisConfig()
    analysis_saving_path = result_saving_path.replace("output", "output/analysis")
    mmva = MultiModalVQAAnalysis(
        evaluation_saving_path=result_saving_path, map_path=analysis_config.map_path
    )
    mmva.analysis(analysis_saving_path=analysis_saving_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default="")
    parser.add_argument("--res-path", type=str, default="")
    parser.add_argument("--level-emb", type=bool, default=False)
    parser.add_argument("--num", type=int, default=-1)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--sheet-names", type=str, nargs="+", default=["CFP"])
    args = parser.parse_args()
    # analysis = MultiModalClassificationAnalysis(args)
    # cm = analysis.calculate_confusion_matrix(args.res_path)
    # print(f"Accuracy: {analysis.calculate_accuracy():.4f}")
    # print(cm)
    analyze_multi_modal_classification()
    # analyze_multi_modal_vqa()
