import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd()))
import json
from fundus_knowledge_base.index_manager.base_index_manager import (
    BaseIndexManager,
    ImageRetrieveResults,
)
from fundus_knowledge_base.data_extractor.multi_disease_data_extractor import (
    MultiDiseaseDataExtractor,
)
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import ImageNode, TextNode
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

class MulitDiseaseConfig:
    default_image_folder = pathlib.Path(
        "./fundus_knowledge_base/img_savings/Classic Images"
    )
    default_image_embedding_saving_folder = pathlib.Path(
        "./fundus_knowledge_base/emb_savings/mulit_disease_clip_embedding"
    )
    default_text_embedding_saving_folder = pathlib.Path("")
    default_model_name: str = "BAAI/bge-small-en-v1.5"


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
            embed_model=HuggingFaceEmbedding(model_name=model_name),
        )

        # save index
        if not saving_folder.exists():
            saving_folder.mkdir()
        self.image_index.storage_context.persist(persist_dir=saving_folder)

    def load_index(self, saving_folder: pathlib.Path, model_name: str = "ViT-B/32"):
        if saving_folder != None:
            if saving_folder.exists():
                storage_context_classic = StorageContext.from_defaults(
                    persist_dir=str(saving_folder),
                )
                multi_index = load_index_from_storage(
                    storage_context_classic,
                    embed_model=HuggingFaceEmbedding(model_name=model_name),
                )
            else:
                print("invalid emb " + saving_folder)
                multi_index = None
        else:
            print("None emb")
            multi_index = None
        return multi_index

    def retrieve_image(self, multi_index, img_path, top_k) -> ImageRetrieveResults:
        txt = []
        score = []
        img = []
        metadata = []
        if multi_index != None:
            retrieve_data = multi_index.as_retriever(
                similarity_top_k=top_k, image_similarity_top_k=top_k
            )
            nodes = retrieve_data.image_to_image_retrieve(img_path)
            for node in nodes:
                txt.append(node.get_text())  # excudates
                score.append(node.get_score())  # 0.628
                img.append(node.node.image_path)
                metadata.append(node.node.metadata)
        # return {"txt": txt, "score": score, "img": img, "metadata": metadata}
        return ImageRetrieveResults(txt=txt, score=score, img=img, metadata=metadata)

