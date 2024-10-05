# 思路
## 问题描述
 - 输入：
    - 一张眼底图
    - 眼底图的seg
 - 输出：
    - 眼底图的诊断结果

## RAG目标
 - 找到Knowledge Base中seg或者原始img最接近的图像作为诊断参考
 - 找到Knowledge Base中文本表示最接近的作为参考

## 步骤
 - 对每个img给出粗略判断
 - 用llama-index构建Knowledge Base
 - 在llava-med中添加增强