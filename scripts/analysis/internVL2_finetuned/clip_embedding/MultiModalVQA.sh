classic_emb=classic_emb_clip
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
model=internVL2_finetuned
output_path="./output/${dataset}/${model}/raw_${m}_${n}_rag_${test_num}_pics3.json"
log_path="./output/${dataset}/${model}/log/raw_${m}_${n}_rag_${test_num}_pics3.log"
res_path="./output/${dataset}/${model}/analysis/raw_${m}_${n}_rag_${test_num}_pics.png"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate internvl_louwei
python analysis_.py \
    --file-path ${output_path} \
    --res-path ${res_path} 


classic_emb=classic_emb_clip
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
model=internVL2_finetuned
output_path="./output/${dataset}/${model}/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.json"
log_path="./output/${dataset}/${model}/log/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.log"
res_path="./output/${dataset}/${model}/analysis/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.png"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate internvl_louwei
python analysis_.py \
    --file-path ${output_path} \
    --res-path ${res_path} 

# filter
classic_emb=classic_emb_mvqa
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
model=internVL2_finetuned
filter=True
save_tmp=${classic_emb}_${m}_${n}_rag_${test_num}_${filter}_pics_2d
output_path="./output/${dataset}/${model}/${save_tmp}.json"
log_path="./output/${dataset}/${model}/log/${save_tmp}.log"
res_path="./output/${dataset}/${model}/analysis/${save_tmp}.png"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate internvl_louwei
python analysis_.py \
    --file-path ${output_path} \
    --res-path ${res_path} 