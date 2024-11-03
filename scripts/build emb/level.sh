python emb_builder.py \
    --img-path "./data/level/" \
    --emb-path "./data/level_emb_clip/"


layer=6
python emb_builder.py \
    --img-path "./data/level/" \
    --emb-path "./data/level_emb_clip_${layer}/" \
    --layer ${layer}

layer=3
python emb_builder.py \
    --img-path "./data/level/" \
    --emb-path "./data/level_emb_clip_${layer}/" \
    --layer ${layer}