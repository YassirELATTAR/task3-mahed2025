#!/bin/bash
# Multimodal Hate Speech Detection - Experiment Runner
# Usage first time: chmod +x run_experiments.sh && ./run_experiments.sh
# Usage again: ./run_experiments.sh
# 
# This script runs two types of experiments:
# 1. Single fusion method (transformer in this example)
# 2. Ensemble of all fusion methods
#
# To modify:
# - Change --data_dir to your dataset path
# - Change --fusion to desired method: concatenate, cross_attention, early, attention_weighted, transformer, late
# - Set --ensemble True to run all methods and ensemble predictions
# - Adjust --result_dir and --model_dir for output paths
# - Modify model names with --text_model and --visual_model

# Run single fusion experiment

python main.py \
    --data_dir "../data/Prop2Hate-Meme" \
    --fusion "transformer" \
    --ensemble False \
    --result_dir "./results" \
    --model_dir "./models" \
    --text_model "UBC-NLP/MARBERTv2" \
    --visual_model "openai/clip-vit-large-patch14" \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --batch_size 16

# Run ensemble experiment
python main.py \
    --data_dir "../data/Prop2Hate-Meme" \
    --ensemble True \
    --result_dir "./results" \
    --model_dir "./models" \
    --text_model "UBC-NLP/MARBERTv2" \
    --visual_model "openai/clip-vit-large-patch14" \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --batch_size 16