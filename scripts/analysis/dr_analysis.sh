# level rag finetuned analysis
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med-finetuned/${level_emb}_${m}_${n}_rag_${test_num}_pics.json"
log_path="./output/${dataset}/llava-med-finetuned/log/${level_emb}_${m}_${n}_rag_${tet_num}_pics.log"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis.py \
    --file-path ${output_path} 