import pandas as pd
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
from wordcloud import WordCloud


@dataclass
class MultiModalVQAAnalysisConfig:
    map_path = "/home/hongyu/Visual-RAG-LLaVA-Med/data/mapping_relationship.xlsx"


class MultiModalVQAAnalysis:
    def __init__(self, evaluation_saving_path, map_path):
        self.res_path = evaluation_saving_path
        self.classes = ["Normal", "Referable DR"]
        if isinstance(self.res_path, str) and self.res_path.endswith("json"):
            with open(self.res_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            self.df = pd.DataFrame(results)
        else:
            self.df = pd.read_csv(self.res_path)
        self.map_path = map_path
        self.mapping_df = pd.read_excel(self.map_path)
        self.seed = 52

    def analysis_modality(self):
        cfp_count_answer, cfp_count_gt, cfp_acc = self.cal_cfp_acc()
        ffa_count_answer, ffa_count_gt, ffa_acc = self.cal_ffa_acc()
        oct_count_answer, oct_count_gt, oct_acc = self.cal_oct_acc()
        avg_modality_acc = (cfp_count_answer + ffa_count_answer + oct_count_answer) / (
            cfp_count_gt + ffa_count_gt + oct_count_gt
        )
        print(f"CFP modality accuracy: {cfp_acc:.2%}")
        print(f"FFA modality accuracy: {ffa_acc:.2%}")
        print(f"OCT modality accuracy: {oct_acc:.2%}")
        print(f"Average modality accuracy: {avg_modality_acc:.2%}")
        return {
            "CFP_acc": cfp_acc,
            "FFA_acc": ffa_acc,
            "OCT_acc": oct_acc,
            "AVG_acc": avg_modality_acc,
        }

    def cal_cfp_acc(self):
        cfp_gt = self.df[self.df["answer"] == "This is a color fundus image."]
        cfp_count_gt = cfp_gt.shape[0]
        cfp_count_answer = (
            cfp_gt["llm respond"].str.contains("color fundus", case=False).sum()
        )
        cfp_accuracy = cfp_count_answer / cfp_count_gt
        return cfp_count_answer, cfp_count_gt, cfp_accuracy

    def cal_ffa_acc(self):
        ffa_gt = self.df[
            self.df["answer"] == "This is a fundus fluorescein angiography (FFA) image."
        ]
        ffa_count_gt = ffa_gt.shape[0]
        ffa_count_answer = (
            ffa_gt["llm respond"]
            .str.contains("fundus fluorescein angiography|FFA", case=False)
            .sum()
        )
        ffa_accuracy = ffa_count_answer / ffa_count_gt
        return ffa_count_answer, ffa_count_gt, ffa_accuracy

    def cal_oct_acc(self):
        oct_gt = self.df[
            self.df["answer"] == "This is an optical coherence tomography (OCT) image."
        ]
        oct_count_gt = oct_gt.shape[0]
        oct_count_answer = (
            oct_gt["llm respond"]
            .str.contains("optical coherence tomography|OCT", case=False)
            .sum()
        )
        oct_accuracy = oct_count_answer / oct_count_gt
        return oct_count_answer, oct_count_gt, oct_accuracy

    def analysis_eye(self):
        left_eye_count_answer, left_eye_count_gt, left_eye_accuracy = (
            self.cal_left_acc()
        )
        right_eye_count_answer, right_eye_count_gt, right_eye_accuracy = (
            self.cal_right_acc()
        )
        average_eye_accuracy = (left_eye_count_answer + right_eye_count_answer) / (
            left_eye_count_gt + right_eye_count_gt
        )
        print(f"Left eye accuracy: {left_eye_accuracy:.2%}")
        print(f"Right eye accuracy: {right_eye_accuracy:.2%}")
        print(f"Average eye accuracy: {average_eye_accuracy:.2%}")
        return {
            "left acc": left_eye_accuracy,
            "right acc": right_eye_accuracy,
            "avg acc": average_eye_accuracy,
        }

    def cal_left_acc(self):
        left_eye_gt = self.df[self.df["answer"] == "Left eye."]
        left_eye_count_gt = left_eye_gt.shape[0]
        left_eye_count_answer = (
            left_eye_gt["llm respond"].str.contains("left eye", case=False).sum()
        )
        left_eye_accuracy = left_eye_count_answer / left_eye_count_gt
        return left_eye_count_answer, left_eye_count_gt, left_eye_accuracy

    def cal_right_acc(self):
        right_eye_gt = self.df[self.df["answer"] == "Right eye."]
        right_eye_count_gt = right_eye_gt.shape[0]
        right_eye_count_answer = (
            right_eye_gt["llm respond"].str.contains("right eye", case=False).sum()
        )
        right_eye_accuracy = right_eye_count_answer / right_eye_count_gt
        return right_eye_count_answer, right_eye_count_gt, right_eye_accuracy

    def analysis_diagnosis(self):
        # Create input to general diagnosis mapping dictionary
        mapping_dict = dict(
            zip(self.mapping_df["input"], self.mapping_df["general diagnosis"])
        )

        # Extract diagnosis
        self.df["diagnosis"] = self.df["answer"].str.extract(
            r"The possible diagnosis of this image is (.+?)\."
        )[0]

        # Calculate correct diagnosis count
        correct_diagnosis_count = 0
        diagnosis_count = 0

        # Iterate through each row, check if answer contains general diagnosis or its reverse mapping in mapping relationship.xlsx
        for index, row in self.df.iterrows():
            # Get corresponding general diagnosis
            general_diagnosis = mapping_dict.get(row["diagnosis"])

            if general_diagnosis is not None:
                diagnosis_count += 1
                # Collect all possible inputs
                possible_inputs = self.mapping_df[
                    self.mapping_df["general diagnosis"] == general_diagnosis
                ]["input"].tolist()

                # Check if answer contains general diagnosis or its corresponding inputs
                if general_diagnosis in row["llm respond"] or any(
                    input_item in row["llm respond"] for input_item in possible_inputs
                ):
                    correct_diagnosis_count += 1

        # Calculate diagnosis accuracy
        diagnosis_accuracy = (
            correct_diagnosis_count / diagnosis_count if diagnosis_count > 0 else 0
        )
        print(f"Diagnosis accuracy: {diagnosis_accuracy:.2%}")
        return diagnosis_accuracy

    def analysis(self, analysis_saving_path):
        modality = self.analysis_modality()
        eye = self.analysis_eye()
        diag = self.analysis_diagnosis()
        fin_res = {"modality": modality, "eye": eye, "diag": diag}
        with open(analysis_saving_path, "w", encoding="utf-8") as f:
            json.dump(fin_res, f)

    def plot_word_cloud_comparsion(
        self,
        wordcloud_saving_path,
        words_to_remove={"image", "appear", "provided", "suggests", "appears"},
        key_word: str = None,
    ):
        gt_text = ""
        pred_text = ""
        for index, row in self.df.iterrows():
            # Get corresponding general diagnosis
            if key_word is None:
                gt_text += row["answer"]
                pred_text += row["llm respond"]
            else:
                if key_word in row["answer"]:
                    gt_text += row["answer"]
                    pred_text += row["llm respond"]
                else:
                    pass
        pred_text_cleaned = " ".join(
            word for word in pred_text.split() if word.lower() not in words_to_remove
        )
        gt_text_cleaned = " ".join(
            word for word in gt_text.split() if word.lower() not in words_to_remove
        )
        wordcloud1 = WordCloud(
            width=800,
            height=400,
            max_words=200,
            background_color="white",
            random_state=self.seed,
        ).generate(pred_text_cleaned)

        # 生成第二个词云
        wordcloud2 = WordCloud(
            width=800,
            height=400,
            max_words=200,
            background_color="white",
            random_state=self.seed,
        ).generate(gt_text_cleaned)

        # 创建图像和轴
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # 在左侧显示第一个词云，并添加标题
        ax1.imshow(wordcloud1, interpolation="bilinear")
        ax1.axis("off")
        ax1.set_title("LLM Respond", fontsize=20)

        # 在右侧显示第二个词云，并添加标题
        ax2.imshow(wordcloud2, interpolation="bilinear")
        ax2.axis("off")
        ax2.set_title("Ground Truth", fontsize=20)

        if key_word is None:
            title = "WordCloud LLM vs GT"
        else:
            title = "WordCloud LLM vs GT" + f" : key word = {key_word}"

        # 调整布局以避免重叠
        plt.tight_layout()
        plt.title(title)
        plt.savefig(wordcloud_saving_path)
