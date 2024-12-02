import os
import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import ast

class BaseAnalysis:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_data()

    def load_data(self):
        with open(self.filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def plot_confusion_matrix(self, cm, classes, res_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
        plt.savefig(res_path)
        plt.show()

    # Additional common methods can be defined here...


class DiabeticRetinopathyAnalysis(BaseAnalysis):
    def __init__(self, filepath, res_path, level_emb=False):
        super().__init__(filepath)
        self.res_path = res_path
        self.level_emb = level_emb
        self.classes = ["Normal", "Referable DR"]

    def analyze(self):
        # 数据预处理：将 mild 和 moderate 类别合并到 Normal 或 Referable DR
        # 计算度量标准
        accuracy, match_rate, error_prob = self.calculate_metrics()
        relationship = self.analyze_relationship(accuracy, match_rate)

        # 绘制混淆矩阵并保存结果
        self.plot_matrices_and_save_results(accuracy, match_rate, error_prob, relationship)

    def calculate_metrics(self, data):
        # # 初始化统计字典
        correct_counts = defaultdict(int)
        total_counts = defaultdict(int)
        match_counts = defaultdict(int)
        error_types = defaultdict(lambda: defaultdict(int))  # 错误类型统计
        
        for item in data["results"]:
            ground_truth = item["ground truth"]
            correct = item["correct"]
            
            # 统计总数量
            total_counts[ground_truth] += 1
            
            # 统计结果正确数量
            if correct:
                correct_counts[ground_truth] += 1
                
            if len(eval(item["record_data"]["ret_l"])["txt"]) != 0:
                predicted = eval(item["record_data"]["ret_l"])["txt"][0]  # 假设返回的第一个结果是预测值
                if predicted != item["ground truth"]:
                    # 如果不正确，统计错误类型
                    error_types[ground_truth][predicted] += 1
                else:
                    pass
            
            # 统计匹配数量
            ret_l_txt = eval(item["record_data"]["ret_l"])["txt"]
            if ret_l_txt and ground_truth in ret_l_txt:
                match_counts[ground_truth] += 1
        
        # 计算正确率和匹配率
        accuracy = {}
        match_rate = {}
        error_probabilities = {}  # 错误概率
        for gt in total_counts:
            accuracy[gt] = correct_counts[gt] / total_counts[gt]
            match_rate[gt] = match_counts[gt] / total_counts[gt]
            
            # 计算错误概率
            error_total = total_counts[gt]   # 错误总数
            if error_total > 0:
                error_probabilities[gt] = {error: count / error_total for error, count in error_types[gt].items()}
            else:
                error_probabilities[gt] = {}  # 没有错误则为空字典
        
        return accuracy, match_rate, error_probabilities

    def analyze_relationship(self, accuracy, match_rate):
        relationship = {}
        for gt in accuracy:
            relationship[gt] = (accuracy[gt], match_rate[gt])
        return relationship

    def plot_matrices_and_save_results(self, accuracy, match_rate, error_prob, relationship):
        # Plot matrices and write results to file...
        pass


class StepwiseAccuracyAnalysis(BaseAnalysis):
    def __init__(self, filepath, step=4):
        super().__init__(filepath)
        self.step = step
        self.classes = ["Normal", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]

    def calculate_accuracy(self):
        # Implement the logic to calculate accuracy at a given step...
        pass

    def save_confusion_matrix(self, res_path, normalize=True):
        # Implement the logic to save confusion matrix...
        pass

    def save_binary_confusion_matrix(self, res_path, normalize=True):
        # Implement the logic to save binary confusion matrix...
        pass

class MultiVQAAnalysis(BaseAnalysis):
    def __init__(self, filepath):
        super().__init__(filepath)

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
    
    def calculate_confusion_matrix(self):
        # 初始化变量用于存储所有的 ground truths 和 predictions
        all_ground_truths = []
        all_predictions = []

        # 获取所有可能的类别
        classes = set()
        
        for result in self.data['results']:
            ground_truth = result['ground truth'].lower()
            try:
                llm_respond = str(json.loads(result['llm respond'])).lower()
            except:
                llm_respond = result["llm respond"].lower()

            # 假设 llm_respond 是一个 JSON 字符串，其中包含 'diagnosis' 键作为预测结果
            # 如果不是这种情况，您可能需要调整如何从 llm_respond 提取预测值
            prediction = llm_respond if ground_truth in llm_respond else "incorrect"

            # 添加到集合中以确保唯一性
            classes.add(ground_truth)
            classes.add(prediction)

            # 将当前的 ground truth 和 prediction 添加到列表中
            all_ground_truths.append(ground_truth)
            all_predictions.append(prediction)

        # 将集合转换为排序后的列表
        classes = sorted(list(classes))
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_ground_truths, all_predictions, labels=classes)

        # 使用父类的方法绘制混淆矩阵
        res_path = 'confusion_matrix.png'
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
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(res_path)
        plt.show()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default="")
    parser.add_argument("--res-path", type=str, default="")
    parser.add_argument("--level-emb", type=bool, default=False)
    parser.add_argument("--num", type=int, default=-1)
    parser.add_argument("--step", type=int, default=4)
    args = parser.parse_args()

    # Determine which analysis to perform based on arguments or other criteria
    # if args.level_emb:
    #     AS = DiabeticRetinopathyAnalysis(args.file_path, args.res_path, args.level_emb)
    #     AS.analyze()
    # else:
    #     AS = StepwiseAccuracyAnalysis(args.file_path, args.step)
    #     print(AS.calculate_accuracy())
    #     AS.save_binary_confusion_matrix(args.res_path)
        
    analysis = MultiVQAAnalysis(args.res_path)
    cm = analysis.calculate_confusion_matrix()
    print(f"Accuracy: {analysis.calculate_accuracy():.4f}")
    print(cm)