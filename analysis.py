import os, json
import argparse
from collections import defaultdict


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
        
        for item in data["results"]:
            ground_truth = item["ground truth"]
            correct = item["correct"]
            
            # 统计总数量
            total_counts[ground_truth] += 1
            
            # 统计正确数量
            if correct:
                correct_counts[ground_truth] += 1
            
            # 统计匹配数量
            ret_l_txt = eval(item["record_data"]["ret_l"])["txt"]
            if ret_l_txt and ground_truth in ret_l_txt:
                match_counts[ground_truth] += 1
        
        # 计算正确率和匹配率
        accuracy = {}
        match_rate = {}
        for gt in total_counts:
            accuracy[gt] = correct_counts[gt] / total_counts[gt]
            match_rate[gt] = match_counts[gt] / total_counts[gt]
        
        return accuracy, match_rate

    def analyze_relationship(self, accuracy, match_rate):
        relationship = {}
        for gt in accuracy:
            relationship[gt] = (accuracy[gt], match_rate[gt])
        return relationship

    def analyze(self,):
        data = self.load_json(self.file_path)
        
        accuracy, match_rate = self.calculate_metrics(data)
        relationship = self.analyze_relationship(accuracy, match_rate)
        
        self.write_results_to_file(accuracy, match_rate, relationship)
            
            
    def write_results_to_file(self, accuracy, match_rate, relationship):
        with open(self.res_path, 'w') as file:
            file.write("Accuracy per ground truth type:\n")
            for gt, acc in accuracy.items():
                file.write(f"{gt}: {acc:.2f}\n")
            
            file.write("\nMatch rate per ground truth type:\n")
            for gt, rate in match_rate.items():
                file.write(f"{gt}: {rate:.2f}\n")
            
            file.write("\nRelationship between accuracy and match rate:\n")
            for gt, (acc, rate) in relationship.items():
                file.write(f"{gt}: Accuracy={acc:.2f}, Match Rate={rate:.2f}\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default="")
    parser.add_argument("--res-path", type=str, default="")
    parser.add_argument("--num", type=int, default=-1)
    args = parser.parse_args()
    AS = Analysis(args)
    AS.analyze()