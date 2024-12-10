from context_former import ContextFormer
class Checker():
    def __init__(self):
        pass
    
    def check_multi_modal_vqa(self, response, ret_cl):
        flag = False
        for i in range(len(ret_cl['txt'])):
            if ret_cl['txt'][i] in response:
                return True, ""
            if ret_cl['score'][i] > 0.7:
                flag=True
        if flag:
            return False, "the diagnosis is not included in matching result, please check the answer"
        else:
            return True, ""
    