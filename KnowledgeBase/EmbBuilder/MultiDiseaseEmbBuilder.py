import sys
sys.path.append(r"/home/hongyu/Visual-RAG-LLaVA-Med/")
import pathlib
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
            representation_file = os.path.join(target_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}.pt")
            torch.save(representation, representation_file)

            # Record the correspondence
            representation_data[image_path] = representation_file

        # Save the correspondence data to a JSON file
        correspondence_file = os.path.join(target_folder, 'correspondence.json')
        with open(correspondence_file, 'w') as f:
            json.dump(representation_data, f)