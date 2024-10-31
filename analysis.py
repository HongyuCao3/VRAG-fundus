import os, json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import ast

class Analysis():
    def __init__(self, args):
        self.file_path = args.file_path
        self.res_path = args.res_path
    
    def load_json(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def calculate_metrics(self, data):
        # 初始化统计字典
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
    
    def analyze(self):
        data = self.load_json(self.file_path)

        # 数据预处理：将 mild 和 moderate 类别合并到 Normal
        for item in data["results"]:
            ground_truth = item["ground truth"]
            if ground_truth in ["mild nonproliferative diabetic retinopathy"]:
                item["ground truth"] = "Normal"

            ret_l_str = item["record_data"]["ret_l"]
            if ret_l_str and ret_l_str.strip():  # 检查 ret_l_str 是否非空
                ret_l = ast.literal_eval(ret_l_str)  # 安全解析字符串为 Python 对象
                if "txt" in ret_l and ret_l["txt"]:
                    predicted = ret_l["txt"][0]
                    if predicted in ["mild nonproliferative diabetic retinopathy"]:
                        ret_l["txt"][0] = "Normal"
                    item["record_data"]["ret_l"] = str(ret_l)  # 将修改后的对象转回字符串

        # 重新计算度量标准
        accuracy, match_rate, error_prob = self.calculate_metrics(data)
        relationship = self.analyze_relationship(accuracy, match_rate)

        y_true, y_pred, classes = self.get_matrix_attr_level_emb(data)

        # 创建混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        # 替换坐标轴标签
        plot_classes = ["Normal", "moderate pdr", "severe npdr", "pdr"]

        # 绘制并保存混淆矩阵
        self.plot_confusion_matrix(cm, plot_classes, normalize=True, title='Normalized Confusion Matrix')

        self.write_results_to_file(accuracy, match_rate, error_prob, relationship)  
    
    def get_matrix_attr_level_emb(self, data):
        y_true = []
        y_pred = []
        for item in data["results"]:
            y_true.append(item["ground truth"])
            ret_l_str = item["record_data"]["ret_l"]
            if ret_l_str and ret_l_str.strip():  # 检查 ret_l_str 是否非空
                ret_l = ast.literal_eval(ret_l_str)  # 安全解析字符串为 Python 对象
                if "txt" in ret_l and ret_l["txt"]:
                    y_pred.append(ret_l["txt"][0])  # 假设返回的第一个结果是预测值
                else:
                    y_pred.append(None)  # 如果没有预测值，则用 None 表示
            else:
                y_pred.append(None)  # 如果 ret_l_str 为空，则用 None 表示

        # 指定类别的顺序
        classes = ["Normal", "moderate nonproliferative diabetic retinopathy", "severe nonproliferative diabetic retinopathy", "proliferative diabetic retinopathy"]
        return y_true, y_pred, classes
    
    def write_results_to_file(self, accuracy, match_rate, error_prob, relationship):
        with open(self.res_path, 'w') as file:
            file.write("Accuracy per ground truth type:\n")
            for gt, acc in accuracy.items():
                file.write(f"{gt}: {acc:.2f}\n")
            
            file.write("\nMatch rate per ground truth type:\n")
            for gt, rate in match_rate.items():
                file.write(f"{gt}: {rate:.2f}\n")
                
            file.write("\nError type and prob:\n")
            for gt, rate in error_prob.items():
                file.write(f"{gt}: \n")
                for k,v in rate.items():
                    # if k == gt:
                    #     continue
                    file.write(f"\t{k}: {v}\n")
            
            file.write("\nRelationship between accuracy and match rate:\n")
            for gt, (acc, rate) in relationship.items():
                file.write(f"{gt}: Accuracy={acc:.2f}, Match Rate={rate:.2f}\n")
            
    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
        plt.savefig(self.res_path.replace('.txt', '_confusion_matrix.png'))
        plt.show()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default="")
    parser.add_argument("--res-path", type=str, default="")
    parser.add_argument("--num", type=int, default=-1)
    args = parser.parse_args()
    AS = Analysis(args)
    AS.analyze()