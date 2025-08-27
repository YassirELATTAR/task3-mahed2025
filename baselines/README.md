## Baselines

This folder contains baseline implementations for multimodal hate speech detection.

### Contents

- `Multimodal_example.ipynb`  
  Original baseline provided by the Mahed2025 team  

- `text_only.py`  
  Text-only baseline using **MARBERTv2** for Arabic text classification  

- `visual_only.py`  
  Image-only baseline using **CLIP** visual encoder  

- `combine_text_image_models.py`  
  Late fusion baseline combining predictions from separate text and image models  


### Usage
Run individual baselines:
```bash
python text_only.py
python visual_only.py  
python combine_text_image_models.py
```

The Jupyter notebook can be run interactively:
```bash
jupyter notebook Multimodal_example.ipynb
```
These baselines serve as comparison points for the advanced fusion methods implemented in the main scripts.