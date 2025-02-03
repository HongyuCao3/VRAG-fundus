import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd()))
from tqdm import tqdm
import json
from Datasets.MultiModalClassificationDataset import (
    MultiModalClassificationDataset,
    MultiModalClassificationConfig,
)
from torch.utils.data import DataLoader
from QwenVLVRAG.QwenVL_vrag import QwenVLVRAG
from PathManager.EmbPathManager import EmbPathManager, EmbPathConfig


class evaluation_config:
    root_dir = pathlib.Path("/home/hongyu/Visual-RAG-LLaVA-Med/QwenVLVRAG/output/")


class evaluation:
    def __init__(self):
        self.config = evaluation_config()

    def evaluate_classification(
        self,
        checkpoint_path: str = "./Model/Qwen2.5-VL-7B-Instruct",
        image_index_folder: pathlib.Path = None,
        text_emb_folder: pathlib.Path= None,
        batch_size: int = 1,
        sheet_names=["CFP"],
        use_pics: bool = False,
        test_num: int=-1,
    ):
        vrag = QwenVLVRAG(checkpoint_path=checkpoint_path)
        dataset_config = MultiModalClassificationConfig()
        dataset = MultiModalClassificationDataset(
            dataset_config.DEFAULT_EXCEL_PATH, sheet_names=sheet_names
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Iterate through the dataloader
        results = []
        query = "what is th diagnosis?"
        for idx, (images, diagnosis) in tqdm(enumerate(dataloader)):
            if test_num != -1 and idx >= test_num:
                break
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": images[0],
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ]
            answer = vrag.inference_rag(
                messages=messages,
                image_index_folder=image_index_folder,
                text_emb_folder=text_emb_folder,
                use_pics=use_pics,
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
            f"{image_index_name}_{text_emb_name}_usepics_{str(use_pics)}.json"
        )
        with result_saving_path.open("w", encoding="utf-8") as f:
            json.dump(results, f)


if __name__ == "__main__":
    eva = evaluation()
    pm = EmbPathManager()
    text_emb_folder = pm.get_emb_dir(pm.config.default_text_emb_name)
    image_index_folder = pathlib.Path(
        "./fundus_knowledge_base/emb_savings/mulit_desease_image_index"
    )
    eva.evaluate_classification(image_index_folder=image_index_folder, text_emb_folder=text_emb_folder, test_num=1)
