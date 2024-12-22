import os
import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class LesionBalancedAnalysis():
    def __init__(self, file_path, res_path):
        self.res_path = res_path
        self.classes = ["Normal", "Referable DR"]
        self.df = pd.read_csv(self.res_path)
    
    def analysis_modality(self):
        cfp_count_answer, cfp_count_gt, cfp_acc = self.cal_cfp_acc()
        ffa_count_answer, ffa_count_gt, ffa_acc = self.cal_ffa_acc()
        oct_count_answer, oct_count_gt, oct_acc = self.cal_oct_acc()
        avg_modality_acc = (cfp_count_answer + ffa_count_answer + oct_count_answer) / (cfp_count_gt + ffa_count_gt + oct_count_gt)
        print(f"CFP modality accuracy: {cfp_acc:.2%}")
        print(f"FFA modality accuracy: {ffa_acc:.2%}")
        print(f"OCT modality accuracy: {oct_acc:.2%}")
        print(f"Average modality accuracy: {avg_modality_acc:.2%}")
        return {"CFP_acc": cfp_acc, "FFA_acc": ffa_acc, "OCT_acc": oct_acc, "AVG_acc": avg_modality_acc}
        
    def cal_cfp_acc(self):
        cfp_gt = self.df[self.df['answer'] == "This is a color fundus image."]
        cfp_count_gt = cfp_gt.shape[0]
        cfp_count_answer = cfp_gt['llm respond'].str.contains("color fundus", case=False).sum()
        cfp_accuracy = cfp_count_answer / cfp_count_gt
        return cfp_count_answer, cfp_count_gt, cfp_accuracy
    
    def cal_ffa_acc(self): 
        ffa_gt = self.df[self.df['answer'] == "This is a fundus fluorescein angiography (FFA) image."]
        ffa_count_gt = ffa_gt.shape[0]
        ffa_count_answer = ffa_gt['llm respond'].str.contains("fundus fluorescein angiography|FFA", case=False).sum()
        ffa_accuracy = ffa_count_answer / ffa_count_gt
        return ffa_count_answer, ffa_count_gt,ffa_accuracy
    
    def cal_oct_acc(self):
        oct_gt = self.df[self.df['answer'] == "This is an optical coherence tomography (OCT) image."]
        oct_count_gt = oct_gt.shape[0]
        oct_count_answer = oct_gt['llm respond'].str.contains("optical coherence tomography|OCT", case=False).sum()
        oct_accuracy = oct_count_answer / oct_count_gt
        return oct_count_answer, oct_count_gt,oct_accuracy
    
    def analysis_eye(self):
        left_eye_count_answer, left_eye_count_gt, left_eye_accuracy =  self.cal_left_acc()
        right_eye_count_answer, right_eye_count_gt, right_eye_accuracy = self.cal_right_acc()
        average_eye_accuracy = (left_eye_count_answer + right_eye_count_answer) / (left_eye_count_gt + right_eye_count_gt)
        print(f"Left eye accuracy: {left_eye_accuracy:.2%}")
        print(f"Right eye accuracy: {right_eye_accuracy:.2%}")
        print(f"Average eye accuracy: {average_eye_accuracy:.2%}")
        return {"left acc": left_eye_accuracy, "right acc": right_eye_accuracy, "avg acc": average_eye_accuracy}
        
    def cal_left_acc(self):
        left_eye_gt = self.df[self.df['answer'] == "Left eye."]
        left_eye_count_gt = left_eye_gt.shape[0]
        left_eye_count_answer = left_eye_gt['llm respond'].str.contains("left eye", case=False).sum()
        left_eye_accuracy = left_eye_count_answer / left_eye_count_gt
        return left_eye_count_answer, left_eye_count_gt, left_eye_accuracy
    
    def cal_right_acc(self):
        right_eye_gt = self.df[self.df['answer'] == "Right eye."]
        right_eye_count_gt = right_eye_gt.shape[0]
        right_eye_count_answer = right_eye_gt['llm respond'].str.contains("right eye", case=False).sum()
        right_eye_accuracy = right_eye_count_answer / right_eye_count_gt
        return right_eye_count_answer, right_eye_count_gt, right_eye_accuracy