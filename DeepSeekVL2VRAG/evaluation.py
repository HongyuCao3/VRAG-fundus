import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd()))
from tqdm import tqdm
import json
from Datasets.MultiModalClassificationDataset import (
    MultiModalClassificationDataset,
    MultiModalClassificationConfig,
)
from Datasets.MultiModalVQADataset import MultiModalVQAConfig, MultiModalVQADataset
from torch.utils.data import DataLoader
from PathManager.EmbPathManager import EmbPathManager, EmbPathConfig
from DeepSeekVL2VRAG.deepseekvl2_vrag import DeepSeekVL2VRAG


class evaluation_config:
    root_dir = pathlib.Path("/home/hongyu/Visual-RAG-LLaVA-Med/DeepSeekVL2VRAG/output/")


class evaluation:
    def __init__(self):
        self.config = evaluation_config()

    def evaluate_classification_prematch(
        self,
        checkpoint_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/deepseek-vl2-small",
        image_index_folder: pathlib.Path = None,
        text_emb_folder: pathlib.Path = None,
        pre_match_results_path: str ="./fundus_knowledge_base/pre_match_savings/classification_cfp.json",
        batch_size: int = 1,
        sheet_names=["CFP"],
        use_pics: bool = False,
        test_num: int = -1,
    ):
        vrag = DeepSeekVL2VRAG(checkpoint_path=checkpoint_path)
        dataset_config = MultiModalClassificationConfig()
        dataset = MultiModalClassificationDataset(
            dataset_config.DEFAULT_EXCEL_PATH, sheet_names=sheet_names
        )
        with open(pre_match_results_path, 'r', encoding="utf-8") as f:
            pre_match_results = json.load(f)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Iterate through the dataloader
        results = []
        query = "what is th diagnosis?"
        for idx, (images, diagnosis) in tqdm(enumerate(dataloader)):
            if test_num != -1 and idx >= test_num:
                break
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n {query}.",
                    "images": [images[0]],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            answer = vrag.inference_rag_prematch(
                conversation=conversation,
                image_index_folder=image_index_folder,
                text_emb_folder=text_emb_folder,
                pre_match_results=pre_match_results[idx]
            )
            results.append(
                {
                    "img_name": images[0],
                    "ground truth": diagnosis[0],
                    "llm respond": answer,
                }
            )
        # save_result
        result_saving_folder = self.config.root_dir / dataset_config.dataset_name
        if not result_saving_folder.exists():
            result_saving_folder.mkdir()
        if image_index_folder:
            image_index_name = image_index_folder.name
        else:
            image_index_name = None
        if text_emb_folder:
            text_emb_name = text_emb_folder.name
        else:
            text_emb_name = None
        result_saving_path = result_saving_folder.joinpath(
            f"{image_index_name}_{text_emb_name}_usepics_{str(use_pics)}_{test_num}.json"
        )
        with result_saving_path.open("w", encoding="utf-8") as f:
            json.dump(results, f)
            
if __name__ == "__main__":
    eva = evaluation()
    pm = EmbPathManager()
    text_emb_folder = pm.get_emb_dir(pm.config.default_text_emb_name)
    # text_emb_folder = None
    # image_index_folder = pathlib.Path(
    #     "./fundus_knowledge_base/emb_savings/mulit_desease_image_index"
    # )
    image_index_folder = None
    eva.evaluate_classification_prematch(
        image_index_folder=image_index_folder,
        text_emb_folder=text_emb_folder,
        test_num=-1,
        use_pics=True,
    )
