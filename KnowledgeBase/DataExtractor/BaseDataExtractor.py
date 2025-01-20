import os
import json
import sys
sys.path.append("/home/hongyu/Visual-RAG-LLaVA-Med")
from abc import ABC
from pathlib import Path
from PathManager.EmbPathManager import EmbPathManager

class BaseDataExtractor(ABC):
    def __init__(self):
        self.path_manager = EmbPathManager()
    
    def extract_image_data(self, folder: str)-> list:
        """get list of (path, text, meta_data) from folder
        args: 
            - folder (str): image forlder, absolute path
        """
        image_data = []

        # 遍历指定目录下的所有JSON文件
        for filename in os.listdir(folder):
            if filename.endswith('.json'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r') as f:
                    # 加载JSON内容
                    data = json.load(f)
                    
                    # 提取信息并构成元组
                    for text, image_path in data.items():
                        image_abs_path = Path.joinpath(folder, image_path)
                        meta_data = {'source_file': image_abs_path}  
                        image_data.append((image_abs_path, text, meta_data))

        return image_data
    
if __name__ == "__main__":
    bde = BaseDataExtractor()
    lesion_image_dir = bde.path_manager.get_image_dir("lesion")
    image_data = bde.extract_image_data(folder=lesion_image_dir)
    print(image_data)