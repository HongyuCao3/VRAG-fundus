import os
import json
from abc import ABC, classmethod
class BaseDataExtractor():
    def __init__(self):
        pass
    
    def extract_image_data(self, folder: str)-> list:
        """get list of (path, text, meta_data) from folder
        args: 
            - folder (str): image forlder, contains a correspondance.json
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
                        meta_data = {'source_file': filename}  # 可以根据需要添加更多元数据
                        image_data.append((image_path, text, meta_data))

        return image_data