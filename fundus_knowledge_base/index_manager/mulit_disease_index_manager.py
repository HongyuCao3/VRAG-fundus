import sys

sys.path.append(r"/home/hongyu/Visual-RAG-LLaVA-Med/")
import json
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from fundus_knowledge_base.index_manager.base_index_manager import BaseIndexManager
from fundus_knowledge_base.data_extractor.multi_disease_data_extractor import (
    MultiDiseaseDataExtractor,
)
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.core.schema import ImageNode, TextNode
import pathlib


class MulitDiseaseConfig:
    default_image_folder = pathlib.Path(
        "./fundus_knowledge_base/img_savings/Classic Images"
    )
    default_image_embedding_saving_folder = pathlib.Path(
        "./fundus_knowledge_base/emb_savings/mulit_disease_clip_embedding"
    )
    default_text_embedding_saving_folder = pathlib.Path("")
    default_model_name: str = "ViT-B/32"


class MultiDiseaseIndexManager(BaseIndexManager):
    def __init__(self):
        super().__init__()
        self.data_extractor = MultiDiseaseDataExtractor()

    def build_image_index(
        self,
        image_folder: pathlib.Path,
        saving_folder: pathlib.Path,
        model_name: str = "ViT-B/32",
    ):
        document = self.data_extractor.extract_image_data(image_folder)
        image_nodes = [
            ImageNode(image_path=str(p), text=t, meta_data=k) for p, t, k in document
        ]

        self.image_index = MultiModalVectorStoreIndex(
            image_nodes,
            show_progress=True,
            embed_model=ClipEmbedding(model_name=model_name),
        )

        # save index
        if not saving_folder.exists():
            saving_folder.mkdir()
        self.image_index.storage_context.persist(persist_dir=saving_folder)

    def build_text_index(
        self,
        text_file: pathlib.Path,
        saving_folder: pathlib.Path,
        model_name: str = "ViT-B/32",
    ):
        with open(text_file, "r", encoding="utf-8") as f:
            text_documents = json.load(f)
        text_nodes = [TextNode(text=f"{k}: {v}") for k, v in text_documents.items()]

        self.text_index = MultiModalVectorStoreIndex(
            text_nodes,
            show_progress=True,
            embed_model=ClipEmbedding(model_name=model_name),
        )

        # save index
        if not saving_folder.exists():
            saving_folder.mkdir()
        self.text_index.storage_context.persist(persist_dir=saving_folder)

    def load_index(self, saving_folder: pathlib.Path, model_name: str = "ViT-B/32"):
        if saving_folder != None:
            if saving_folder.exists():
                storage_context_classic = StorageContext.from_defaults(
                    persist_dir=str(saving_folder),
                )
                multi_index = load_index_from_storage(storage_context_classic,  embed_model=ClipEmbedding(model_name=model_name))
            else:
                print("invalid emb " + saving_folder)
                multi_index = None
        else:
            print("None emb")
            multi_index = None
        return multi_index
    
    def retrieve(self, multi_index, img_path, top_k):
        txt = []
        score = [] 
        img = [] 
        metadata = []
        if multi_index != None:
            retrieve_data = multi_index.as_retriever(similarity_top_k=top_k, image_similarity_top_k=top_k)
            nodes = retrieve_data.image_to_image_retrieve(img_path)
            for node in nodes:
                txt.append(node.get_text()) # excudates
                score.append(node.get_score()) # 0.628
                img.append(node.node.image_path)
                metadata.append(node.node.metadata)
            # TODO: add text retrieve
        return {"txt": txt, "score": score, "img": img, "metadata": metadata}
    
if __name__ == "__main__":
    mdim = MultiDiseaseIndexManager()
    text_file = pathlib.Path("./data/Classic Images/classic.json",)
    text_index_saving_folder = pathlib.Path("./fundus_knowledge_base/emb_savings/mulit_desease_text_index")
    image_index_saving_folder = pathlib.Path("./fundus_knowledge_base/emb_savings/mulit_desease_image_index")
    
    # build index
    # mdim.build_text_index(text_file="./data/Classic Images/classic.json", saving_folder=text_index_saving_folder)
    # mdim.build_image_index(image_folder=pathlib.Path("./data/Classic Images/"), aving_folder=image_index_saving_folder)
    
    # image retrieve
    image_index = mdim.load_index(saving_folder=image_index_saving_folder)
    image_path = "./data/Classic Images/DR/mild NPDR_1.jpeg"
    result = mdim.retrieve(multi_index=image_index, img_path=image_path, top_k=3)
    print(result)
    
    # text retrieve
    # text_index = mdim.load_index(saving_folder=saving_folder)
    # retrieve_data = text_index.as_retriever(similarity_top_k=1, image_similarity_top_k=3)
    # node = retrieve_data.retrieve
    
