# Models

This folder stores trained model checkpoints from experiments.

## Structure

Model checkpoints are saved with the naming convention:
```
best_model_{fusion_type}_{loss_type}.pth
```

Where:
- `fusion_type`: concatenate, cross_attention, early, attention_weighted, transformer, late
- `loss_type`: weighted, focal

## Usage

Models are automatically saved during training and loaded for evaluation. The best performing model for each configuration is preserved based on validation macro F1 score.

To load a specific model:
```python
model = MultimodalClassifier(fusion_type='transformer', args=args)
model.load_state_dict(torch.load('best_model_transformer_focal.pth'))
```