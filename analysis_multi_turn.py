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
        self.classes = ["Normal", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]

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
    
    def save_confusion_matrix(self, step, res_path, normalize=True):
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
            if not convert_full_name_to_abbreviation(llm_level) in self.classes:
                print(llm_level)
            if not convert_full_name_to_abbreviation(ground_truth) in self.classes:
                print(ground_truth)
            y_true.append(convert_full_name_to_abbreviation(ground_truth))
            y_pred.append(convert_full_name_to_abbreviation(llm_level))

        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
     
        self.plot_confusion_matrix(cm, self.classes, res_path=res_path, normalize=True)
    
    def plot_confusion_matrix(self, cm, classes, res_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # 保存图像
        plt.savefig(res_path)
        plt.show()
    
    def save_binary_confusion_matrix(self, step, res_path, normalize=True):
        """绘制并保存二元混淆矩阵"""
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
            
            # 合并类别
            binary_ground_truth = 'Normal' if convert_full_name_to_abbreviation(ground_truth) in ['Normal', 'Mild NPDR'] else 'Referable DR'
            binary_llm_level = 'Normal' if convert_full_name_to_abbreviation(llm_level) in ['Normal', 'Mild NPDR'] else 'Referable DR'

            y_true.append(binary_ground_truth)
            y_pred.append(binary_llm_level)

        cm = confusion_matrix(y_true, y_pred, labels=['Normal', 'Referable DR'])
     
        self.plot_confusion_matrix(cm, ['Normal', 'Referable DR'], res_path=res_path, normalize=normalize)
        
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
    AS.save_binary_confusion_matrix(args.step, args.res_path)