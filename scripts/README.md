# Scripts

This folder contains the main training infrastructure and model implementations.

## Contents

- **`main.py`**: Main training script with all fusion methods and ensemble techniques
- **`config.py`**: Configuration management and argument parsing
- **`dataset.py`**: Multimodal dataset class for loading text and image data
- **`run_experiments.sh`**: Shell script to run experiments with predefined configurations

## Key Features

The main script implements:
- 6 fusion methods: concatenate, cross_attention, early, attention_weighted, transformer, late
- Ensemble predictions with majority vote and weighted averaging  
- Focal loss and weighted cross-entropy for imbalanced data
- Support for different text and visual encoders
- Comprehensive logging and checkpointing

## Usage

Run single experiment:
```bash
python main.py --fusion transformer --data_dir ../data
```

Run ensemble:
```bash  
python main.py --ensemble True --data_dir ../data
```

Use the shell script for batch experiments:
```bash
./run_experiments.sh
```