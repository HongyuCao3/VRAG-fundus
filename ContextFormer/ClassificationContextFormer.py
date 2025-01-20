import sys
sys.path.append("/home/hongyu/Visual-RAG-LLaVA-Med")
from ContextFormer.BaseContextFormer import BaseContextFormer, BaseContextConfig
class ClassificationContextConfig(BaseContextConfig):
    DR_diagnosing_level = {"Normal": "No lesion","Mild NPDR": "MAs only", "Moderate NPDR": "At least one hemorrhage or MA and/or at least one of the following: Retinal hemorrhages, Hard exudates, Cotton wool spots, Venous beading", "Severe NPDR": "Any of the following but no signs of PDR (4-2-1 rule): >20 intraretinal hemorrhages in each of four quadrants, definite venous, beading in two or more quadrants, Prominent IRMA in one or more quadrants", "PDR": "One of either: Neovascularization, Vitreous/preretinal hemorrhage"}
    DR_level_mapping = {
            "Normal": "Normal",
            "Mild NPDR": "mild nonproliferative diabetic retinopathy",
            "Moderate NPDR": "moderate nonproliferative diabetic retinopathy",
            "Severe NPDR": "severe nonproliferative diabetic retinopathy",
            "PDR": "proliferative diabetic retinopathy"
        }
    lesion = {"microaneurysm": "", "hemorrhage": "", "cotton wool spots": "", "exudates": ""}

class ClassificationContextFormer(BaseContextFormer):
    def __init__(self):
        super().__init__()
        self.config = ClassificationContextConfig()
        
        