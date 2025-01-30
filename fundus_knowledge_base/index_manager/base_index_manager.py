import os, pathlib
from PIL import Image
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageNode

AVAILABLE_CLIP_MODELS = (
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64",
    "ViT-B/32",
    "ViT-B/16",
    "ViT-L/14",
    "ViT-L/14@336px",
)

class BaseIndexManager():
    def __init__(self):
        pass
    
    def extract_image_data_classic(self, folder):
        image_data = []
        # 支持的图片格式列表
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        
        # 获取根目录的绝对路径
        base_dir = os.path.abspath(folder)
        
        for root, dirs, files in os.walk(folder):
            for file in files:
                # 获取文件扩展名
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    # 构建完整的文件路径
                    full_path = os.path.join(root, file)
                    
                    # 获取子文件夹名称
                    sub_folder = os.path.relpath(root, base_dir)
                    
                    # 如果子文件夹是根目录，则显示为空字符串
                    if sub_folder == '.':
                        sub_folder = ''
                    if sub_folder == "DR":
                        text = file.split("_")[0]
                        if text == "no DR":
                            text = "Normal"
                    elif sub_folder == "metaPM":
                        text = file.split("_")[0]
                    else:
                        text = sub_folder
                    image_path = full_path
                    meta_data = file
                    image_data.append((image_path, text, meta_data))
                    print(f"Image found: {file} in subfolder: {sub_folder} at path: {full_path}")
        return image_data
    
    def build_index(self, image_folder: pathlib.Path, saving_folder: pathlib.Path, model_name: str="ViT-B/32"):
        document = self.extract_image_data_classic(image_folder)
        image_nodes = [ImageNode(image_path=p, text=t, meta_data=k) for p, t, k in document]
        
        self.multi_index = MultiModalVectorStoreIndex(image_nodes, show_progress=True, embed_model=ClipEmbedding(model_name=model_name))
        
         # save index
        if not saving_folder.exists():
            saving_folder.mkdir()
        self.multi_index.storage_context.persist(persist_dir=saving_folder)