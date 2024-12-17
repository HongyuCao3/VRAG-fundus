import os
import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import ast
from AnalysisVRAG.base import BaseAnalysis

class MultiVQAAnalysis(BaseAnalysis):
    def __init__(self, args):
        super().__init__(args.filepath)
        self.sheet_names = args.sheet_names

    def calculate_accuracy(self):
        correct_count = 0
        total_count = len(self.data['results'])

        for result in self.data['results']:
            ground_truth = result['ground truth'].lower()
            try:
                llm_respond = str(json.loads(result['llm respond'])).lower()
            except:
                llm_respond = result["llm respond"].lower()
            # print(type(llm_respond))
            # diagnosis = llm_respond.get('diagnosis', '').lower()

            if ground_truth in llm_respond:
                correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        return accuracy
    
    def calculate_confusion_matrix(self, res_path):
        # 初始化变量用于存储所有的 ground truths 和 predictions
        all_ground_truths = []
        all_predictions = []

        # 获取所有可能的类别
        classes = []
        if "CFP" in self.sheet_names:
            classes = set(['diabetic retinopathy',
            'age-related macular degeneration',
            'central retinal vein occlusion',
            'branch retinal vein occlusion',
            'central retinal artery occlusion',
            'branch retinal artery occlusion',
            'central serous chorioretinopathy',
            'retinal detachment',
            'Coats Disease',
            'macular hole',
            'pathologic myopia',
            'glaucoma',
            'epiretinal membrane'])
        if "FFA" in self.sheet_names:
            classes.append(["diabetic retinopathy", 
                            "wet age-related macular degeneration", 
                            "dry age-related macular degeneration", 
                            "central retinal vein occlusion", 
                            "branch retinal vein occlusion", 
                            "central serous chorioretinopathy",
                            "choroidal melanoma", 
                            "Coats Disease", 
                            "familial exudative vitreoretinopathy", 
                            "Vogt-Koyanagi-Harada disease"])
            classes = set(classes)
            
        if "OCT" in self.sheet_names:
            classes.append(["cystoid macular edema", 
                            "central serous chorioretinopathy",
                            "dry age-related macular degeneration",
                            "epiretinal membrane" "macular hole",
                            "polypoidal choroidal vasculopathy",
                            "retinal detachment",
                            "retinoschisis",
                            "retinal vein occlusion",
                            "wet age-related macular degeneration"])
            classes = set(classes)
        for result in self.data['results']:
            ground_truth = result['ground truth'].lower()
            try:
                llm_respond = str(json.loads(result['llm respond'])).lower()
            except:
                llm_respond = result["llm respond"].lower()

            # 假设 llm_respond 是一个 JSON 字符串，其中包含 'diagnosis' 键作为预测结果
            # 如果不是这种情况，您可能需要调整如何从 llm_respond 提取预测值
            if ground_truth in llm_respond:
                prediction = ground_truth 
            else:
                for c in classes:
                     if c in llm_respond:
                         prediction = c
                if prediction is None:
                    prediction = "incorrect"

            # 添加到集合中以确保唯一性
            # classes.add(ground_truth)
            # classes.add(prediction)

            # 将当前的 ground truth 和 prediction 添加到列表中
            all_ground_truths.append(ground_truth.lower())
            all_predictions.append(prediction.lower())

        # 将集合转换为排序后的列表
        classes = sorted(list(classes))
        # print(classes)
        # 计算混淆矩阵
        cm = confusion_matrix(all_ground_truths, all_predictions, labels=classes)

        # 使用父类的方法绘制混淆矩阵
        self.plot_confusion_matrix(cm, classes, res_path, normalize=True, title='Normalized Confusion Matrix')
        
    def plot_confusion_matrix(self, cm, classes, res_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, annotate=False):
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
        
        if annotate:  # Check if annotation is needed
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i,
                        format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        font_size = 6  # 您可以更改这个值以适应您的需求
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontdict={'fontsize': font_size})
        plt.yticks(tick_marks, classes, fontdict={'fontsize': font_size})
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(res_path)
        plt.show()