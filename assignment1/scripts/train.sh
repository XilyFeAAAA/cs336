##! /bin/bash

# Pretrain the model
python ../train_transformer.py  \
        --batch_size 20 \
        --bpe_fp data/TinyStoriesV2-GPT4-valid.txt \
        --train_dataset_fp data/TinyStoriesV2-GPT4-train.txt \
        --validate_dataset_fp data/TinyStoriesV2-GPT4-valid.txt \
        --save_fp data


