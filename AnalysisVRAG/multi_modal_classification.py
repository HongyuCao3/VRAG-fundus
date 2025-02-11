import os, sys

sys.path.append("/home/hongyu/Visual-RAG-LLaVA-Med")
import json
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from wordcloud import WordCloud
from AnalysisVRAG.base import BaseAnalysis
from AnalysisVRAG.class_combiner import ClassCombiner


class MultiModalClassificationAnalysis(BaseAnalysis):
    def __init__(self, file_path, sheet_names):
        super().__init__(file_path)
        self.sheet_names = sheet_names
        self.combiner = ClassCombiner()
        self.seed = 52

    def calculate_confusion_matrix(self, image_saving_path):
        # 初始化变量用于存储所有的 ground truths 和 predictions
        all_ground_truths = []
        all_predictions = []

        # 获取所有可能的类别
        classes = set([])
        if "CFP" in self.sheet_names:
            classes = set(
                [
                    "diabetic retinopathy",
                    "age-related macular degeneration",
                    # 'central retinal vein occlusion',
                    # 'branch retinal vein occlusion',
                    "retinal vein occlusion",
                    # 'central retinal artery occlusion',
                    # 'branch retinal artery occlusion',
                    "retinal artery occlusion",
                    "central serous chorioretinopathy",
                    "retinal detachment",
                    "coats disease",
                    "macular hole",
                    "pathologic myopia",
                    "glaucoma",
                    "epiretinal membrane",
                ]
            )
        if "FFA" in self.sheet_names:
            classes.update(
                {
                    "diabetic retinopathy",
                    # "wet age-related macular degeneration",
                    # "dry age-related macular degeneration",
                    "age-related macular degeneration",
                    # "central retinal vein occlusion",
                    # "branch retinal vein occlusion",
                    "retinal vein occlusion" "central serous chorioretinopathy",
                    "choroidal melanoma",
                    "coats disease",
                    "familial exudative vitreoretinopathy",
                    "vogt-koyanagi-harada disease",
                }
            )
            # classes = set(classes)

        if "OCT" in self.sheet_names:
            classes.update(
                {
                    "cystoid macular edema",
                    "central serous chorioretinopathy",
                    # "dry age-related macular degeneration",
                    "age-related macular degeneration",
                    "epiretinal membrane",
                    "macular hole",
                    "polypoidal choroidal vasculopathy",
                    "retinal detachment",
                    "retinoschisis",
                    "retinal vein occlusion",
                    # "wet age-related macular degeneration"
                }
            )
            # classes = set(classes)
        print(self.sheet_names)
        if isinstance(self.data, list):
            data = copy.deepcopy(self.data)
            self.data = {"results": data}
        acc = 0
        for result in self.data["results"]:
            ground_truth = result["ground truth"].lower()
            try:
                llm_respond = str(json.loads(result["llm respond"])).lower()
            except:
                if isinstance(result["llm respond"], str):
                    llm_respond = result["llm respond"].lower()
                if isinstance(result["llm respond"], list):
                    llm_respond = result["llm respond"][0].lower()

            ground_truth = self.combiner.combine(ground_truth)
            # 假设 llm_respond 是一个 JSON 字符串，其中包含 'diagnosis' 键作为预测结果
            # 如果不是这种情况，您可能需要调整如何从 llm_respond 提取预测值
            if ground_truth in llm_respond:
                prediction = ground_truth
                acc += 1
            else:
                flag = False
                for c in classes:
                    if c.lower() in llm_respond:
                        prediction = c.lower()
                        flag = True
                if not flag:
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

        accuracy = acc / len(self.data["results"])

        # 使用父类的方法绘制混淆矩阵
        self.plot_confusion_matrix(
            cm,
            classes,
            image_saving_path,
            normalize=True,
            title="Normalized Confusion Matrix",
            accuracy=accuracy,
        )

        return accuracy

    def plot_confusion_matrix(
        self,
        cm,
        classes,
        image_saving_path,
        normalize=False,
        title="Confusion matrix",
        cmap=plt.cm.Blues,
        annotate=False,
        accuracy=None,
    ):
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")
        print(cm)

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if annotate:  # Check if annotation is needed
            fmt = ".2f" if normalize else "d"
            thresh = cm.max() / 2.0
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        font_size = 6  # 您可以更改这个值以适应您的需求
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontdict={"fontsize": font_size})
        plt.yticks(tick_marks, classes, fontdict={"fontsize": font_size})
        if accuracy is not None:
            plt.text(
                cm.shape[0],
                -1,
                f"Accuracy: {accuracy:.2f}",
                color="red",
                horizontalalignment="left",
                verticalalignment="top",
            )
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(image_saving_path)
        plt.show()

    def plot_word_cloud_comparsion(
        self,
        wordcloud_saving_path,
        words_to_remove={"image", "appear", "provided", "suggests", "appears"},
    ):
        gt_text = ""
        pred_text = ""

        if isinstance(self.data, list):
            data = copy.deepcopy(self.data)
            self.data = {"results": data}
        for result in self.data["results"]:
            ground_truth = result["ground truth"].lower()
            try:
                llm_respond = str(json.loads(result["llm respond"])).lower()
            except:
                if isinstance(result["llm respond"], str):
                    llm_respond = result["llm respond"].lower()
                if isinstance(result["llm respond"], list):
                    llm_respond = result["llm respond"][0].lower()

            ground_truth = self.combiner.combine(ground_truth)
            pred_text += llm_respond
            gt_text += ground_truth

        pred_text_cleaned = " ".join(
            word for word in pred_text.split() if word.lower() not in words_to_remove
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
        ).generate(gt_text)

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

        # 调整布局以避免重叠
        plt.tight_layout()
        plt.savefig(wordcloud_saving_path)
