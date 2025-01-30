import sys

sys.path.append(r"/home/hongyu/Visual-RAG-LLaVA-Med/")
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from fundus_knowledge_base.index_manager.base_index_manager import BaseIndexManager
from fundus_knowledge_base.data_extractor.multi_disease_data_extractor import MultiDiseaseDataExtractor
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.core.schema import ImageNode
import pathlib

class MulitDiseaseConfig():
    default_image_folder = pathlib.Path("./fundus_knowledge_base/img_savings/Classic Images")
    default_image_embedding_saving_folder = pathlib.Path("./fundus_knowledge_base/emb_savings/mulit_disease_clip_embedding")
    default_text_embedding_saving_folder = pathlib.Path("")
    default_model_name: str="ViT-B/32"
    


class MultiDiseaseIndexManager(BaseIndexManager):
    def __init__(self):
        super().__init__()
        self.data_extractor = MultiDiseaseDataExtractor()
        
    
    def build_index(self, image_folder: pathlib.Path, saving_folder: pathlib.Path, model_name: str="ViT-B/32"):
        document = self.data_extractor.extract_image_data(image_folder)
        image_nodes = [ImageNode(image_path=p, text=t, meta_data=k) for p, t, k in document]
        
        self.multi_index = MultiModalVectorStoreIndex(image_nodes, show_progress=True, embed_model=ClipEmbedding(model_name=model_name))
        
         # save index
        if not saving_folder.exists():
            saving_folder.mkdir()
        self.multi_index.storage_context.persist(persist_dir=saving_folder)
        
    def load_index(self, saving_folder: pathlib.Path):
        if saving_folder != None:
            if saving_folder.exists():
                storage_context_classic = StorageContext.from_defaults(persist_dir=saving_folder)
                multi_index = load_index_from_storage(storage_context_classic)
            else:
                print("invalid emb "+saving_folder)
                multi_index=None
        else:
            print("None emb")
            multi_index=None
        return multi_index
        