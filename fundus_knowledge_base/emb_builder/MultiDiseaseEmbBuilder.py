import sys
sys.path.append(r"/home/hongyu/Visual-RAG-LLaVA-Med/")
import pathlib
import torch
import json
from transformers import CLIPModel, CLIPProcessor
from fundus_knowledge_base.emb_builder.BaseEmbBuilder import BaseEmbBuilder
from PathManager.EmbPathManager import EmbPathManager
from Datasets.MultiModalClassificationDataset import MultiModalClassificationConfig
from fundus_knowledge_base.data_extractor.multi_disease_data_extractor import MultiDiseaseDataExtractor

class MultiDiseaseEmbBuilder(BaseEmbBuilder):
    def __init__(self, model_name: str="openai/clip-vit-base-patch32"):
        super().__init__(model_name=model_name)
        self.default_text_emb_name = "MultiDiseaseText"
        self.default_img_emb_name = "MultiDiseaseImage"
        self.default_img_dir_name = "Classic Images"
        self.path_manager = EmbPathManager()
        self.data_extractor = MultiDiseaseDataExtractor()
        
    def save_image_representaion(self, source_folder: pathlib.Path, target_folder: pathlib.Path, layer_index=11):
        if not target_folder.exists():
            target_folder.mkdir()
        image_data = self.data_extractor.extract_image_data(folder=source_folder)
        representation_data = {}
        for image_path, disease, meta_data in image_data:
            
            representation = self.get_image_embedding(image_path, layer_index)
            
            # Save the representation as a .pt file
            representation_path = pathlib.Path.joinpath(target_folder, f"{image_path.stem}.pt")
            torch.save(representation, representation_path)

            # Record the correspondence
            representation_data[image_path] = representation_path

        # Save the correspondence data to a JSON file
        correspondence_file = pathlib.Path.joinpath(target_folder, 'correspondence.json')
        with open(correspondence_file, 'w') as f:
            json.dump(representation_data, f)
            
if __name__ == "__main__":
    config = MultiModalClassificationConfig()
    path_manager = EmbPathManager()
    mde = MultiDiseaseEmbBuilder()
    discription_path = pathlib.Path.joinpath(path_manager.get_image_dir("Classic Images"), "classic.json")
    with open(discription_path, "r", encoding="utf-8") as ds_f:
        discription = json.load(ds_f)
    print(discription)
    save_dir = pathlib.Path.joinpath(path_manager.get_emb_dir(path_manager.config.default_text_emb_name))
    print(save_dir)
    mde.build_text_embedings(discription=discription, save_dir=save_dir)
    
    