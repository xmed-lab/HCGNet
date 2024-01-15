cfg_path=./tasks/pdb_search/configs/multiscale+chem.json
tag=hcgnet
gpu=0

save_dir=logs/search_$tag
mkdir $save_dir

CUDA_VISIBLE_DEVICES=$gpu python -u tasks/pdb_search/train.py \
    --batch_size 8 \
    --num_workers 4 \
    --cfg_path $cfg_path \
    --optim sgd \
    --lr 0.01 \
    --max_epoch 150 \
    --ins_norm \
    --forward_single \
    --aug_scale 0.2 \
    --aug_jit_std 0.01 \
    --aug_jit_clip 0.1 \
    --save_dir $save_dir \
