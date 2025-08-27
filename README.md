# Multimodal Hate Speech Detection for Arabic Memes
## Abstract
We present our system for the MAHED 2025 Shared Task on Arabic Hate Meme Detection (subtask 3), a binary classification task to determine whether a multimodal meme containing Arabic text and an image conveys a hateful message. Our approach uses multimodal fusion combining a visual encoder and an Arabic text encoder. We explored four fusion strategies—transformer fusion, early fusion, cross-attention, and bilinear fusion—and found transformer fusion offered the best single-model trade-off, while an ensemble of all four achieved the highest score. To address the severe class imbalance (90.05% not-hate vs. 9.95% hate), we applied class-weighted loss, focal loss, strong regularization, and light augmentation. Our best submission reached a macro-F1 score of **0.75** on the test set.
- **Paper:** [Link to be provided after review] (#)
- **Shared Task:** [MAHED 2025](https://marsadlab.github.io/mahed2025/)
## Repository Structure
```bash
├── baselines/                 # Baseline models and experiments
├── data/                     # Data processing and augmentation scripts
├── models/                   # Saved model checkpoints
├── outputs/                  # Model predictions on test sets
├── results/                  # Evaluation reports and results
├── scripts/                  # Main training scripts and configurations
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation
#### 1. Clone the repository:
```bash
git clone git@github.com:YassirELATTAR/task3-mahed2025.git
cd task3-mahed2025
```
#### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

#### 3. Download the dataset:
```bash
cd data
python data_dowload.py
```


## Quick Start

### 1. Data Preparation
Download and analyze the dataset:
```bash
cd data
jupyter notebook data_prep.ipynb
```

### 2. Run Main Experiments
Run single fusion method:
```bash
cd scripts
./run_experiments.sh
```

- Or run with custom configuration:
```bash
bashpython main.py \
    --data_dir "../data/Prop2Hate-Meme" \
    --fusion "transformer" \
    --result_dir "./results" \
    --model_dir "./models"

```
- Run ensemble experiments:
```bash
bashpython main.py \
    --data_dir "../data/Prop2Hate-Meme" \
    --ensemble True \
    --result_dir "./results" \
    --model_dir "./models"
```

### 3. Run Baseline Models
- Execute baseline experiments:
```bash
cd baselines
python text_only.py
python visual_only.py
python combine_text_image_models.py
```
- Original baseline:
```bash
python Multimodal_example.ipynb
```
### 4. Data Augmentation
Run data augmentation scripts:
```bash
cd data
python hate_data_aug.py
python image_text_data_aug.py
python tweet_normalizer.py
```

## Configuration Options

The main script accepts the following arguments:

- `--data_dir`: Path to dataset directory  
- `--fusion`: Fusion method (`concatenate`, `cross_attention`, `early`, `attention_weighted`, `transformer`, `late`)  
- `--ensemble`: Run ensemble of all fusion methods (`True`/`False`)  
- `--text_model`: Text encoder model name (default: `UBC-NLP/MARBERTv2`)  
- `--visual_model`: Visual encoder model name (default: `openai/clip-vit-large-patch14`)  
- `--result_dir`: Output directory for results  
- `--model_dir`: Directory for model checkpoints  
- `--learning_rate`: Learning rate (default: `2e-5`)  
- `--num_epochs`: Number of training epochs (default: `10`)  
- `--batch_size`: Batch size (default: `16`)  

---

## Repository Components

- `/baselines`  
  Contains baseline implementations including text-only, image-only, and combined prediction models, plus the original Mahed2025 baseline.  

- `/data`  
  Data processing utilities including download scripts, augmentation techniques, and normalization tools for both text and image modalities.  

- `/models`  
  Storage for trained model checkpoints organized by fusion method and loss type.  

- `/outputs`  
  Prediction files for both test and gold test splits, organized by model configuration.  

- `/results`  
  Evaluation reports and performance metrics in text format for easy comparison across methods.  

- `/scripts`  
  Main training infrastructure including model definitions, fusion techniques, configuration management, and experiment runners.  


## Results
Performance of fusion approaches on the test set.  
**Macro F1\*** = official submission score on the gold test set used for leaderboard ranking.

| Fusion          | Acc. | Macro F1 | Macro F1* |
|-----------------|:----:|:--------:|:---------:|
| **Ensemble (All)** | 0.90 | 0.72 | **0.75** |
| Transformer     | 0.91 | 0.72 | 0.75 |
| Concatenation   | 0.89 | 0.74 | 0.73 |
| Cross-Attn.     | 0.88 | 0.69 | 0.68 |
| Bilinear        | 0.89 | 0.63 | 0.66 |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{elattar-2025-yassirea,
  author    = {El Attar, Yassir},
  title     = {YassirEA at MAHED 2025: Fusion-Based Multimodal Models for Arabic Hate Meme Detection},
  booktitle = {Proceedings of the Third Arabic Natural Language Processing Conference (ArabicNLP 2025)},
  month     = nov,
  year      = {2025},
  address   = {Suzhou, China},
  publisher = {Association for Computational Linguistics}
}
```

## Contact

For questions or issues, please contact:
Yassir El Attar via email yassir.el.attar@gmail.com


