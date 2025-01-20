import sys
sys.path.append(r"/home/hongyu/Visual-RAG-LLaVA-Med/")
import pathlib
import torch
import json
from KnowledgeBase.EmbBuilder.BaseEmbBuilder import BaseEmbBuilder
from PathManager.EmbPathManager import EmbPathManager
from KnowledgeBase.DataExtractor.MultiDiseaseDataExtractor import MultiDiseaseDataExtractor

class MultiDiseaseEmbBuilder(BaseEmbBuilder):
    def __init__(self):
        super().__init__()
        self.path_manager = EmbPathManager()
        self.data_extractor = MultiDiseaseDataExtractor()
        
    def save_image_representaion(self, source_folder: pathlib.Path, target_folder: pathlib.Path, layer_index=11):
        if not target_folder.exists():
            target_folder.mkdir()
        image_data = self.data_extractor.extract_image_data(folder=source_folder)
        representation_data = {}
        for image_path, disease, meta_data in image_data:
            
            representation = self.get_layer_representation(image_path, layer_index)
            
            # Save the representation as a .pt file
            representation_path = pathlib.Path.joinpath(target_folder, f"{image_path.stem}.pt")
            torch.save(representation, representation_path)

            # Record the correspondence
            representation_data[image_path] = representation_path

        # Save the correspondence data to a JSON file
        correspondence_file = pathlib.Path.joinpath(target_folder, 'correspondence.json')
        with open(correspondence_file, 'w') as f:
            json.dump(representation_data, f)
            
    # TODO: 添加text representation
            
if __name__ == "__main__":
    mde = MultiDiseaseEmbBuilder()
    