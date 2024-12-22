from context_former import ContextFormer

class VRAGFilter():
    def __init__(self, context_former, threshold=0.5, sheet_names=["CFP"]):
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
        self.threshold = threshold
    
    def filter_multi_modal_vqa(self, ret_cl):
        # 如果是finetune存在的就不用rag
        # print(ret_cl["txt"][0])
        flag=False
        for i in range(len(ret_cl['txt'])):
            if ret_cl['txt'][i] in self.multi_modal_vqa_classes and ret_cl['score'][i] >= self.threshold:
                flag=True
        if not flag:
            return self.context_former.ret_empty
        else:
            return ret_cl