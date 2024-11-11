import os, json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import itertools
import ast
from utils import find_longest_matching_class, convert_abbreviation_to_full_name, convert_full_name_to_abbreviation

class Analysis:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_data()
        # self.classes = list(convert_abbreviation_to_full_name.__closure__[0].cell_contents.keys())
        self.classes = ["Normal", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
        # self.accuracy = self.calculate_accuracy()

    def load_data(self):
        """加载JSON文件中的数据"""
        with open(self.filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def calculate_accuracy(self, step):
        """计算准确率"""
        correct_count = 0
        total_count = len(self.data['results'])

        for result in tqdm(self.data['results']):
            ground_truth = result['ground truth']
            try:
                if step == 4:
                    llm_respond = eval(result['llm respond'])
                else:
                    llm_respond = eval(result["record_data"][f"step {step}"]["response"])
                llm_level = convert_abbreviation_to_full_name(llm_respond['level'])
            except:
                if step == 4:
                    llm_respond = result['llm respond']
                else:
                    llm_respond = result["record_data"][f"step {step}"]["response"]
                llm_level = find_longest_matching_class(llm_respond, self.classes)
                llm_level = convert_abbreviation_to_full_name(llm_level)
            if ground_truth == llm_level:
                correct_count += 1
            
        return correct_count / total_count if total_count > 0 else 0
    
    def plot_confusion_matrix(self, step, res_path):
        """绘制并保存混淆矩阵"""
        y_true = []
        y_pred = []

        for result in tqdm(self.data['results']):
            ground_truth = result['ground truth']
            try:
                if step == 4:
                    llm_respond = eval(result['llm respond'])
                else:
                    llm_respond = eval(result["record_data"][f"step {step}"]["response"])
                llm_level = convert_abbreviation_to_full_name(llm_respond['level'])
            except:
                if step == 4:
                    llm_respond = result['llm respond']
                else:
                    llm_respond = result["record_data"][f"step {step}"]["response"]
                llm_level = find_longest_matching_class(llm_respond, self.classes)
                llm_level = convert_abbreviation_to_full_name(llm_level)
            
            y_true.append(convert_full_name_to_abbreviation(ground_truth))
            y_pred.append(convert_full_name_to_abbreviation(llm_level))

        cm = confusion_matrix(y_true, y_pred, labels=self.classes)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.classes, yticklabels=self.classes,
               title="Confusion Matrix",
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # Save the figure to the specified path
        fig.tight_layout()
        plt.savefig(res_path)
        print(f"Confusion matrix saved to {res_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default="")
    parser.add_argument("--res-path", type=str, default="")
    parser.add_argument("--level-emb", type=bool, default=False)
    parser.add_argument("--num", type=int, default=-1)
    parser.add_argument("--step", type=int, default=4)
    args = parser.parse_args()
    AS = Analysis(args.file_path)
    print(AS.calculate_accuracy(args.step))
    AS.plot_confusion_matrix(args.step, args.res_path)