save_dir=logs/search_hcgnet
epoch=106
gpu=0

CUDA_VISIBLE_DEVICES=$gpu python -u tasks/pdb_search/train.py \
    --test \
    --ins_norm \
    --save_dir $save_dir \
    --resume_epoch $epoch
