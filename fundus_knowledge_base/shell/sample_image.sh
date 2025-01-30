# multi vqa ffa
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
tgt_dir=${cur_path}"/KnowledgeBase/img_savings/Classic\Images\(FFA)"
python ./KnowledgeBase/image_samplar.py \
    --tgt-dir ${tgt_dir} \
    --diag-list "diabetic retinopathy" "wet age-related macular degeneration" "dry age-related macular degeneration" central retinal vein occlusion" branch retinal vein occlusion" "central serous chorioretinopathy" "choroidal melanoma" "Coats Disease" "familial exudative vitreoretinopathy" "Vogt-Koyanagi-Harada disease"

# multi vqa OCT
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
tgt_dir=${cur_path}"/KnowledgeBase/img_savings/Classic\Images\(OCT)"
python ./KnowledgeBase/image_samplar.py \
    --tgt-dir ${tgt_dir} \
    --diag-list "cystoid macular edema" "central serous chorioretinopathy" "dry age-related macular degeneration" "epiretinal membrane" "macular hole" "polypoidal choroidal vasculopathy" "retinal detachment" "retinoschisis" "retinal vein occlusion" "wet age-related macular degeneration"