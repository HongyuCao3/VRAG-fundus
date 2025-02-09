import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd()))
from fundus_knowledge_base.index_manager.mulit_disease_index_manager import ImageRetrieveResults
from fundus_knowledge_base.knowledge_retriever.TextRetriever import TextRetrieveResults

class VRAGFilter():
    def __init__(self, context_former, image_threshold=0.5, text_threshold=0.5, sheet_names=["CFP"]):
        self.multi_modal_vqa_classes = []
        if "CFP" in sheet_names or "New" in sheet_names:
            self.multi_modal_vqa_classes.append("macular hole")
        if "FFA" in sheet_names or "New" in sheet_names:
            self.multi_modal_vqa_classes.extend(["branch retinal vein occlusion", "Coats Disease", "familial exudative vitreoretinopathy", "Vogt-Koyanagi-Harada disease"])
        if "OCT" in sheet_names or "New" in sheet_names:
            self.multi_modal_vqa_classes.extend(["crystoid macular edema", "polypoidal choroidal vasculopathy", "retinal detachment", "retinoschisis"])
        # self.multi_modal_vqa_classes = [
        # 'Coats Disease',
        # 'macular hole', 
        # 'central serous chorioretinopathy'
        # ]
        self.context_former = context_former
        self.image_threshold = image_threshold
        self.text_threshold = text_threshold
    
    def filter_multi_modal_vqa(self, ret_cl):
        # 如果是finetune存在的就不用rag
        # print(ret_cl["txt"][0])
        flag=False
        for i in range(len(ret_cl['txt'])):
            if ret_cl['txt'][i] in self.multi_modal_vqa_classes and ret_cl['score'][i] >= self.image_threshold:
                flag=True
        if not flag:
            return self.context_former.ret_empty
        else:
            return ret_cl
        
        
    def filter_retrieved_images(self, retrieved_images: ImageRetrieveResults):
        flag= False
        for i in range(len(retrieved_images.txt)):
            if retrieved_images.txt[i] in self.multi_modal_vqa_classes and retrieved_images.score[i] >= self.image_threshold:
                flag=True
        if flag:
            return retrieved_images
        else:
            return ImageRetrieveResults.create_with_empty_lists()
        
    def filter_retrieved_texts(self, retrieved_texts: TextRetrieveResults):
        flag= False
        for i in range(len(retrieved_texts.txt)):
            if retrieved_texts.txt[i] in self.multi_modal_vqa_classes and retrieved_texts.score[i] >= self.image_threshold:
                flag=True
        if flag:
            return retrieved_texts
        else:
            return TextRetrieveResults.create_with_empty_lists()