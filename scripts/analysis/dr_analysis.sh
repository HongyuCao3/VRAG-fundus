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
res_path="./output/${dataset}/llava-med-finetuned/analysis/${level_emb}_${m}_${n}_rag_${tet_num}_pics.txt"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis.py \
    --file-path ${output_path} \
    --res-path ${res_path}

# level rotation rag finetuned analysis
crop_emb=emb_crop
level_emb=level_emb_r
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med-finetuned/${level_emb}_${m}_${n}_rag_${test_num}_pics.json"
res_path="./output/${dataset}/llava-med-finetuned/analysis/${level_emb}_${m}_${n}_rag_${tet_num}_pics.txt"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis.py \
    --file-path ${output_path} \
    --res-path ${res_path}


# raw finetuned analysis
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med-finetuned/raw_${m}_${n}_${test_num}.json"
res_path="./output/${dataset}/llava-med-finetuned/analysis/raw_${m}_${n}_${tet_num}_pics.txt"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis.py \
    --file-path ${output_path} \
    --res-path ${res_path}


# classic dr finetuned analysis
crop_emb=emb_crop
level_emb=classic_dr_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med-finetuned/${level_emb}_${m}_${n}_rag_${test_num}_pics.json"
res_path="./output/${dataset}/llava-med-finetuned/analysis/${level_emb}_${m}_${n}_${test_num}_pics.txt"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis.py \
    --file-path ${output_path} \
    --res-path ${res_path}


# emb finetuned analysis
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med-finetuned/${crop_emb}_${m}_${n}_rag_${test_num}.json"
res_path="./output/${dataset}/llava-med-finetuned/analysis/${crop_emb}_${m}_${n}_${test_num}_pics.txt"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis.py \
    --file-path ${output_path} \
    --res-path ${res_path}

# emb finetuned analysis
crop_emb=emb_crop
level_emb=classic_dr_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=2
n=2
test_num=-1
output_path="./output/${dataset}/llava-med-finetuned/${crop_emb}_${m}_${n}_rag_${test_num}_pics.json"
res_path="./output/${dataset}/llava-med-finetuned/analysis/${crop_emb}_${m}_${n}_rag_${test_num}_pics.txt"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis.py \
    --file-path ${output_path} \
    --res-path ${res_path}