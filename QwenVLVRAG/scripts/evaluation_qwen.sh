export CUDA_VISIBLE_DEVICES=0,1,2,3

conda activate qwenvl
cd /home/hongyu/Visual-RAG-LLaVA-Med
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
log_path=${cur_path}"/QwenVLVRAG/output/log/image.log"
nohup python ./QwenVLVRAG/evaluation.py >$log_path 2>&1 &