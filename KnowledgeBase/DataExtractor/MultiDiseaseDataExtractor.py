import sys

sys.path.append(r"/home/hongyu/Visual-RAG-LLaVA-Med/")
import os
import pathlib
from KnowledgeBase.DataExtractor.BaseDataExtractor import BaseDataExtractor



class MultiDiseaseDataExtractor(BaseDataExtractor):
    def __init__(self):
        super().__init__()
        self.image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]

    def extract_image_data(self, folder: pathlib.Path) -> list:
        image_data = []

        for sub_folder in folder.iterdir():
            for file_name in sub_folder.glob("*"):
                if sub_folder.name in ["DR", "meta PM"]:
                    disease = file_name.name.split("_")[0]
                    if disease == "no DR":
                        disease = "Normal"
                else:
                    disease = sub_folder.name
                image_path = pathlib.Path.joinpath(folder, sub_folder, file_name)
                meta_data = {"source_file": image_path}
                print(
                    f"Image found: {file_name.name} in subfolder: {sub_folder.name} at folder: {folder}"
                )
                image_data.append((image_path, disease, meta_data))
        return image_data
    
if __name__ == "__main__":
    mdde = MultiDiseaseDataExtractor()
    multi_desease_dir = mdde.path_manager.get_image_dir("Classic Images")
    image_data = mdde.extract_image_data(folder=multi_desease_dir)
    print(len(image_data))
