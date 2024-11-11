import os, json
from utils import find_longest_matching_class, convert_abbreviation_to_full_name

class Analysis:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_data()
        self.classes = list(convert_abbreviation_to_full_name.__closure__[0].cell_contents.keys())
        self.accuracy = self.calculate_accuracy()

    def load_data(self):
        """加载JSON文件中的数据"""
        with open(self.filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def calculate_accuracy(self):
        """计算准确率"""
        correct_count = 0
        total_count = len(self.data['results'])

        for result in self.data['results']:
            ground_truth = result['ground truth']
            llm_respond = json.loads(result['llm respond'])
            llm_level = convert_abbreviation_to_full_name(llm_respond['level'])
            if ground_truth == llm_level:
                correct_count += 1
            
        return correct_count / total_count if total_count > 0 else 0

    def get_accuracy(self):
        """获取准确率"""
        return self.accuracy