export CUDA_VISIBLE_DEVICES=0,1,2,3

conda activate deepseekvl2
cd /home/hongyu/Visual-RAG-LLaVA-Med
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
log_path=${cur_path}"/DeepSeekVL2VRAG/output/log/text_vqa.log"
nohup python ./DeepSeekVL2VRAG/evaluation.py >$log_path 2>&1 &