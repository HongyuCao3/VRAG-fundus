export CUDA_VISIBLE_DEVICES=0,1,2,3

cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
log_path=${cur_path}"/QwenVLVRAG/output/log/image_text.log"
nohup python ./QwenVLVRAG/evaluation.py >$log_path 2>&1 &