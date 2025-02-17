import sys, pathlib

sys.path.append(str(pathlib.Path.cwd()))
from tqdm import tqdm
from fundus_knowledge_base.index_manager.mulit_disease_index_manager import (
    MultiDiseaseIndexManager,
)
import json
from Datasets.MultiModalVQADataset import MultiModalVQAConfig, MultiModalVQADataset
from PathManager.EmbPathManager import EmbPathManager, EmbPathConfig
from fundus_knowledge_base.knowledge_retriever.TextRetriever import TextRetriever
from torch.utils.data import DataLoader
from fundus_knowledge_base.pre_match.multi_modal_classification import MultiModalClassificationPreMatch


class MultiModalVQAPreMatch(MultiModalClassificationPreMatch):
    def __init__(self):
        super().__init__()

    def get_vqa_matching_results(
        self,
        image_index_folder: pathlib.Path = None,
        text_emb_folder: pathlib.Path = None,
        batch_size: int = 1,
        sheet_names=["CFP"],
        test_num: int = -1,
        saving_path: pathlib.Path = "./fundus_knowledge_base/pre_match_savings/vqa_cfp.json",
    ):
        dataset_config = MultiModalVQAConfig()
        dataset = MultiModalVQADataset(
            excel_file=dataset_config.DEFAULT_EXCEL_PATH,
            sheet_names=sheet_names,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Iterate through the dataloader
        results = []
        query = "what is th diagnosis?"
        for idx, (images, diagnosis, query, answer) in tqdm(enumerate(dataloader)):
            if test_num != -1 and idx >= test_num:
                break
            matching_result = self.get_matching_results(
                image_path=images[0],
                query=query,
                image_index_folder=image_index_folder,
                text_emb_folder=text_emb_folder,
            )
            results.append(matching_result)
        with open(saving_path, "w", encoding="utf-8") as f:
            json.dump(results, f)

    def get_matching_results(
        self,
        image_path,
        query,
        image_index_folder: pathlib.Path = None,
        text_emb_folder: pathlib.Path = None,
    ):
        if image_index_folder:
            self.index_manager = MultiDiseaseIndexManager()
            self.image_index = self.index_manager.load_index(image_index_folder)
        if text_emb_folder:
            self.text_embedding = TextRetriever(emb_folder=text_emb_folder)
        if image_index_folder:
            retrieved_images = self.index_manager.retrieve_image(
                self.image_index, img_path=image_path, top_k=1
            )
        if text_emb_folder:
            retrieved_texts = self.text_embedding.retrieve(input_img=image_path)

        # form prompt
        if image_index_folder:
            image_context = " ".join(
                [
                    f"{txt}: {score}"
                    for txt, score in zip(retrieved_images.txt, retrieved_images.score)
                ]
            )
        else:
            image_context = None
        if text_emb_folder:
            text_context = " ".join(
                [
                    f"{txt}: {img}"
                    for txt, img in zip(retrieved_texts.txt, retrieved_texts.score)
                ]
            )
        else:
            text_context = None
        prompt = self.build_prompt(
            query=query, image_context=image_context, text_context=text_context
        )
        return {
            "prompt": prompt,
            "text_context": text_context,
            "image_context": image_context,
            "matched_images": retrieved_images.img,
            "src_image": image_path,
        }


if __name__ == "__main__":
    pm = EmbPathManager()
    text_emb_folder = pm.get_emb_dir(pm.config.default_text_emb_name)
    image_index_folder = pathlib.Path(
        "./fundus_knowledge_base/emb_savings/mulit_desease_image_index"
    )
    mmcpm = MultiModalVQAPreMatch()
    mmcpm.get_vqa_matching_results(image_index_folder=image_index_folder, text_emb_folder=text_emb_folder, test_num=-1)
