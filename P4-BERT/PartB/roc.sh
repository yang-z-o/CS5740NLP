#!/bin/bash

export MODEL_FILE=~/Desktop/5740NLP/projects/vp4/mini
export DATA=~/Desktop/5740NLP/projects/vp4/data
export TEST_FILE=~/Desktop/5740NLP/projects/vp4/dev.sav

python3 ./examples/run_roc.py \
    --model_type bert \
    --model_name_or_path output-roberta \
    --output_dir=traningacc \
    --data_dir=$DATA \
    --task_name roc \
    --do_eval
    
    # --do_train \
    # --evaluate_during_training \
    # --do_test

