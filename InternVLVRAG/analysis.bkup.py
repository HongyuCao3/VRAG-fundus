import os
import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
class ClassMap():
    def __init__(self):
        self.multi_modal_classification_map = {
        'diabetic retinopathy': 'diabetic retinopathy',
        'wet age-related macular degeneration': 'age-related macular degeneration',
        'dry age-related macular degeneration': 'age-related macular degeneration',
        'central retinal vein occlusion': 'retinal vein occlusion',
        'branch retinal vein occlusion': 'retinal vein occlusion',
        'central retinal artery occlusion': 'retinal artery occlusion',
        'branch retinal artery occlusion': 'retinal artery occlusion',
        'central serous chorioretinopathy': 'central serous chorioretinopathy',
        'retinal detachment': 'retinal detachment',
        'Coats Disease' : 'Coats Disease',
        'macular hole': 'macular hole',
        'pathologic myopia': 'pathologic myopia',
        'glaucoma': 'glaucoma',
        'epiretinal membrane': 'epiretinal membrane'
        }
    def to_general(self, ground_truth: str):
        if ground_truth in self.multi_modal_classification_map.keys():
            return self.multi_modal_classification_map[ground_truth]
        else:
            return ground_truth
        
class LesionBalancedAnalysis():
    def __init__(self, res_path, map_path, save_path):
        self.res_path = res_path
        self.classes = ["Normal", "Referable DR"]
        self.df = pd.read_csv(self.res_path)
        self.map_path = map_path
        self.save_path = save_path
        self.mapping_df = pd.read_excel(self.map_path)
    
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
    
    def analysis_diagnosis(self):
        # Create input to general diagnosis mapping dictionary
        mapping_dict = dict(zip(self.mapping_df['input'], self.mapping_df['general diagnosis']))

        # Extract diagnosis
        self.df['diagnosis'] = self.df['answer'].str.extract(r'The possible diagnosis of this image is (.+?)\.')[0]

        # Calculate correct diagnosis count
        correct_diagnosis_count = 0
        diagnosis_count = 0

        # Iterate through each row, check if answer contains general diagnosis or its reverse mapping in mapping relationship.xlsx
        for index, row in self.df.iterrows():
            # Get corresponding general diagnosis
            general_diagnosis = mapping_dict.get(row['diagnosis'])

            if general_diagnosis is not None:
                diagnosis_count += 1
                # Collect all possible inputs
                possible_inputs = self.mapping_df[self.mapping_df['general diagnosis'] == general_diagnosis]['input'].tolist()

                # Check if answer contains general diagnosis or its corresponding inputs
                if general_diagnosis in row['llm respond'] or any(input_item in row['llm respond'] for input_item in possible_inputs):
                    correct_diagnosis_count += 1

        # Calculate diagnosis accuracy
        diagnosis_accuracy = correct_diagnosis_count / diagnosis_count if diagnosis_count > 0 else 0
        print(f"Diagnosis accuracy: {diagnosis_accuracy:.2%}")
        return diagnosis_accuracy
    
    def analysis(self):
        modality = self.analysis_modality()
        eye = self.analysis_eye()
        diag = self.analysis_diagnosis()
        fin_res = {"modality": modality, "eye": eye, "diag": diag}
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(fin_res, f)
            
class MultiModalClassificationAnalysis():
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_data()
        self.mapping = ClassMap()
        
    def load_data(self):
        with open(self.filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def calculate_accuracy(self):
        correct_count = 0
        total_count = len(self.data['results'])

        for result in self.data['results']:
            ground_truth = result['ground truth'].lower()
            ground_truth = self.mapping.to_general(ground_truth).lower()
            try:
                llm_respond = str(json.loads(result['llm respond'])).lower()
            except:
                llm_respond = result["llm respond"].lower()
            # print(type(llm_respond))
            # diagnosis = llm_respond.get('diagnosis', '').lower()
            # print(llm_respond)
            if ground_truth in llm_respond:
                correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        return accuracy
    
    def calculate_confusion_matrix(self, res_path):
        # 初始化变量用于存储所有的 ground truths 和 predictions
        all_ground_truths = []
        all_predictions = []

        # 获取所有可能的类别
        classes = set(['diabetic retinopathy',
        'age-related macular degeneration',
        # 'central retinal vein occlusion',
        # 'branch retinal vein occlusion',
        'retinal vein occlusion'
        # 'central retinal artery occlusion',
        # 'branch retinal artery occlusion',
        'retinal artery occlusion'
        'central serous chorioretinopathy',
        'retinal detachment',
        'Coats Disease',
        'macular hole',
        'pathologic myopia',
        'glaucoma',
        'epiretinal membrane'])
        # classes = set(['age-related macular degeneration', 'branch retinal artery occlusion', 'branch retinal vein occlusion', 'central retinal artery occlusion', 'central retinal vein occlusion', 'central serous chorioretinopathy', 'choroidal melanoma', 'coats disease', 'diabetic retinopathy', 'dry age-related macular degeneration', 'epiretinal membrane', 'familial exudative vitreoretinopathy', 'glaucoma', 'macular hole', 'pathologic myopia', 'retinal detachment', 'retinal vein occlusion', 'vogt-koyanagi-harada disease', 'wet age-related macular degeneration'])
        for result in self.data['results']:
            ground_truth = result['ground truth'].lower()
            # ground_truth = self.mapping.to_general(ground_truth)
            try:
                llm_respond = str(json.loads(result['llm respond'])).lower()
            except:
                llm_respond = result["llm respond"].lower()
            # 假设 llm_respond 是一个 JSON 字符串，其中包含 'diagnosis' 键作为预测结果
            # 如果不是这种情况，您可能需要调整如何从 llm_respond 提取预测值
            prediction = None
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-path", type=str,default="")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--map-path", type=str, default='mapping relationship.xlsx')
    args = parser.parse_args()
    lba = LesionBalancedAnalysis(args.res_path, args.map_path, args.save_path)
    lba.analysis()
    # mca = MultiModalClassificationAnalysis(args.res_path)
    # acc = mca.calculate_accuracy()
    # print(acc)
    # mca.calculate_confusion_matrix(args.save_path)