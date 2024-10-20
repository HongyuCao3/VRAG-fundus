# 思路
## 问题描述
 - 输入：
    - 一张眼底图
 - 输出：
    - 眼底图的诊断结果

## RAG目标
 - 根据现有的病灶分割图进行颜色和形态的匹配
 - 找到病灶的文字描述作为增强
 - 使用chunk判断病灶的数量关系

## 步骤
 - 需要测试topk对于结果的影响
 - 需要考虑是否要将所有图片都作为输入
 - 需要测试crop的color超参数对于数据的影响
 - 对于level-emb只能用top1
 - level-emb和crop-emb的context必须分开

## 实验
 - 对比rag与非rag
 - 对比llava和llava-med
 - 对比使用不同knowledge base的方法