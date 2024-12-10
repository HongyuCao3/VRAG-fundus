from context_former import ContextFormer

class VRAGFilter():
    def __init__(self, context_former):
        self.multi_modal_vqa_classes = ['Coats Disease',
        'macular hole', 'central serous chorioretinopathy']
        self.context_former = context_former
    
    def filter_multi_modal_vqa(self, ret_cl):
        # 如果是finetune存在的就不用rag
        if ret_cl['txt'][0] not in self.multi_modal_vqa_classes:
            return self.context_former.ret_empty
        else:
            return ret_cl