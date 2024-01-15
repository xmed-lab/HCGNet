save_dir=logs/site_hcgnet
epoch=146
gpu=0

CUDA_VISIBLE_DEVICES=$gpu python -u tasks/pdb_site/train.py \
    --test \
    --vote 5 \
    --ins_norm \
    --save_dir $save_dir \
    --resume_epoch $epoch
