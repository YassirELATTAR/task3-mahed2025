## Data

This folder contains data processing, augmentation, and analysis utilities.

### Contents

- `data_prep.ipynb`  
  Jupyter notebook for data download, analysis, and gold test evaluation  

- `hate_data_aug.py`  
  Data augmentation specifically for hate speech samples  

- `image_text_data_aug.py`  
  Combined image and text augmentation techniques  

- `tweet_normalizer.py`  
  Arabic text normalization and preprocessing utilities  

- `samples/`  
  Samples to check the predictions to the true labels (human check)

---

### Usage

Start with data preparation:  

```bash
jupyter notebook data_prep.ipynb
```
Run augmentation scripts:
```bash
python hate_data_aug.py
python image_text_data_aug.py
python tweet_normalizer.py
```