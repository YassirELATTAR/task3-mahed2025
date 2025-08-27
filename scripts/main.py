import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, CLIPVisionModel, CLIPProcessor
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging
import torch.nn.functional as F
import numpy as np
from collections import Counter
from config import get_config, save_config
from multimodal_dataset import MultimodalDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create results directory
os.makedirs('../new_results', exist_ok=True)
os.makedirs('../models', exist_ok=True)
os.makedirs('../results/training_logs', exist_ok=True)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


# Enhanced Fusion techniques
class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim, vision_dim, hidden_dim=512):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, text_features, vision_features):
        text_proj = self.text_proj(text_features).unsqueeze(1)
        vision_proj = self.vision_proj(vision_features).unsqueeze(1)
        
        attended_features, _ = self.cross_attention(text_proj, vision_proj, vision_proj)
        fused_features = self.layer_norm(attended_features + text_proj)
        return self.dropout(fused_features.squeeze(1))

class EarlyFusion(nn.Module):
    def __init__(self, text_dim, vision_dim, hidden_dim=1024):
        super().__init__()
        self.fusion_layers = nn.Sequential(
            nn.Linear(text_dim + vision_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(), 
            nn.Dropout(0.2)
        )
    
    def forward(self, text_features, vision_features):
        combined = torch.cat([text_features, vision_features], dim=1)
        return self.fusion_layers(combined)

class LateFusion(nn.Module):
    def __init__(self, text_dim, vision_dim, num_classes=2):
        super().__init__()
        self.text_classifier = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.vision_classifier = nn.Sequential(
            nn.Linear(vision_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, text_features, vision_features):
        text_logits = self.text_classifier(text_features)
        vision_logits = self.vision_classifier(vision_features)
        return (text_logits + vision_logits) / 2

class AttentionWeightedFusion(nn.Module):
    def __init__(self, text_dim, vision_dim, hidden_dim=512):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, text_features, vision_features):
        text_proj = self.text_proj(text_features)
        vision_proj = self.vision_proj(vision_features)
        
        combined = torch.cat([text_proj, vision_proj], dim=1)
        weights = self.attention(combined)
        
        weighted_text = text_proj * weights[:, 0:1]
        weighted_vision = vision_proj * weights[:, 1:2]
        
        return weighted_text + weighted_vision

# New Transformer Fusion
class TransformerFusion(nn.Module):
    def __init__(self, text_dim, vision_dim, hidden_dim=512, num_heads=8, num_layers=2):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.final_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, text_features, vision_features):
        batch_size = text_features.size(0)
        
        text_proj = self.text_proj(text_features).unsqueeze(1)  # [B, 1, H]
        vision_proj = self.vision_proj(vision_features).unsqueeze(1)  # [B, 1, H]
        
        # Combine text and vision features
        combined = torch.cat([text_proj, vision_proj], dim=1)  # [B, 2, H]
        combined = combined + self.pos_embedding
        
        # Apply transformer
        transformed = self.transformer(combined)  # [B, 2, H]
        transformed = self.layer_norm(transformed)
        
        # Global representation
        fused = transformed.view(batch_size, -1)  # [B, 2*H]
        output = self.final_proj(fused)  # [B, H]
        
        return output


class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes=2, fusion_type='concatenate', args=None):
        super(MultimodalClassifier, self).__init__()
        
        # MARBERTv2 for text
        #self.bert = AutoModel.from_pretrained('UBC-NLP/MARBERTv2')
        self.bert = AutoModel.from_pretrained(args.text_model)
        
        self.bert_drop = nn.Dropout(0.3)
        text_dim = self.bert.config.hidden_size  # 768
        
        # CLIP Vision for images
        #self.clip_vision = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')
        self.clip_vision = CLIPVisionModel.from_pretrained(args.visual_model)
        vision_dim = self.clip_vision.config.hidden_size  # 768
        
        self.fusion_type = fusion_type
        
        # Fusion layers
        if fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(text_dim, vision_dim)
            fusion_dim = 512
        elif fusion_type == 'early':
            self.fusion = EarlyFusion(text_dim, vision_dim)
            fusion_dim = 512
        elif fusion_type == 'late':
            self.fusion = LateFusion(text_dim, vision_dim, num_classes)
            fusion_dim = None  # Late fusion outputs directly
        elif fusion_type == 'attention_weighted':
            self.fusion = AttentionWeightedFusion(text_dim, vision_dim)
            fusion_dim = 512
        elif fusion_type == 'transformer':
            self.fusion = TransformerFusion(text_dim, vision_dim)
            fusion_dim = 512
        else:  # concatenate
            self.fusion = None
            fusion_dim = text_dim + vision_dim
        
        # Output layers
        if fusion_type != 'late':
            self.bert_fc = nn.Linear(text_dim, 512)
            self.clip_fc = nn.Linear(vision_dim, 512)
            
            if fusion_type == 'concatenate':
                self.fusion_fc = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
            else:
                self.fusion_fc = nn.Sequential(
                    nn.Linear(fusion_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                
            self.output_fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

    def forward(self, text, image, mask):
        # Text processing with MARBERTv2
        bert_output = self.bert(text, attention_mask=mask, return_dict=False)
        text_features = self.bert_drop(bert_output[0][:, 0, :])  # Use [CLS] token
        
        # Image processing with CLIP
        vision_output = self.clip_vision(pixel_values=image)
        vision_features = vision_output.pooler_output
        
        # Apply fusion
        if self.fusion_type == 'late':
            return self.fusion(text_features, vision_features)
        elif self.fusion_type == 'concatenate':
            text_proj = self.bert_fc(text_features)
            vision_proj = self.clip_fc(vision_features)
            features = torch.cat((text_proj, vision_proj), dim=1)
            features = self.fusion_fc(features)
        else:
            # Other fusion methods including transformer
            fused_features = self.fusion(text_features, vision_features)
            features = self.fusion_fc(fused_features)
        
        output = self.output_fc(features)
        return output

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []
    
    for data in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        text = data["text"].to(device)
        image = data["img_path"].to(device)
        mask = data["text_mask"].to(device)
        labels = data['hate_label'].to(device)
        
        output = model(text, image, mask)
        loss = criterion(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    train_loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return train_loss, accuracy, f1, macro_f1

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            text = data["text"].to(device)
            image = data["img_path"].to(device)
            mask = data["text_mask"].to(device)
            labels = data['hate_label'].to(device)
            
            output = model(text, image, mask)
            loss = criterion(output, labels)
            
            probs = F.softmax(output, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return test_loss, accuracy, f1, macro_f1, all_preds, all_labels, all_probs

class EnsembleClassifier:
    def __init__(self, models, fusion_types, device):
        self.models = models
        self.fusion_types = fusion_types
        self.device = device
        
    def predict_majority_vote(self, test_loader):
        all_predictions = []
        
        for model in self.models:
            model.eval()
            preds = []
            with torch.no_grad():
                for data in test_loader:
                    text = data["text"].to(self.device)
                    image = data["img_path"].to(self.device)
                    mask = data["text_mask"].to(self.device)
                    
                    output = model(text, image, mask)
                    _, predicted = torch.max(output, 1)
                    preds.extend(predicted.cpu().numpy())
            all_predictions.append(preds)
        
        # Majority vote
        ensemble_preds = []
        for i in range(len(all_predictions[0])):
            votes = [pred_list[i] for pred_list in all_predictions]
            ensemble_preds.append(Counter(votes).most_common(1)[0][0])
        
        return ensemble_preds
    
    def predict_weighted_vote(self, test_loader, weights):
        all_probs = []
        
        for model in self.models:
            model.eval()
            probs = []
            with torch.no_grad():
                for data in test_loader:
                    text = data["text"].to(self.device)
                    image = data["img_path"].to(self.device)
                    mask = data["text_mask"].to(self.device)
                    
                    output = model(text, image, mask)
                    prob = F.softmax(output, dim=1)
                    probs.extend(prob.cpu().numpy())
            all_probs.append(np.array(probs))
        
        # Weighted average
        weighted_probs = np.zeros_like(all_probs[0])
        for i, (probs, weight) in enumerate(zip(all_probs, weights)):
            weighted_probs += probs * weight
        
        ensemble_preds = np.argmax(weighted_probs, axis=1)
        return ensemble_preds.tolist()
    
    def predict_transformer_weighted(self, test_loader):
        # Weight transformer fusion higher
        weights = []
        for fusion_type in self.fusion_types:
            if fusion_type == 'transformer':
                weights.append(0.4)
            elif fusion_type == 'early':
                weights.append(0.3)
            else:
                weights.append(0.3 / (len(self.fusion_types) - 2))
        
        return self.predict_weighted_vote(test_loader, weights)

def save_epoch_results(results, fusion_type, loss_type, filename_prefix, args=None):
    """Save training results across epochs"""
    df = pd.DataFrame(results)
    if args:
        filename = os.path.join(args.result_dir, 'training_logs', f'{filename_prefix}_{fusion_type}_{loss_type}_epochs.csv')
    else:
        filename = f'../results/training_logs/{filename_prefix}_{fusion_type}_{loss_type}_epochs.csv'
    df.to_csv(filename, index=False)
    logger.info(f"Epoch results saved to {filename}")

def save_final_results(all_results, loss_type, args=None):
    """Save final results for all fusion methods"""
    results_df = pd.DataFrame(all_results).T
    if args:
        filename = os.path.join(args.result_dir, f'final_results_{loss_type}.csv')
    else:
        filename = f'../results/final_results_{loss_type}.csv'
    results_df.to_csv(filename)
    logger.info(f"Final results saved to {filename}")

def read_jsonl_to_df(filename):
    return pd.read_json(filename, lines=True)

def prepare_dataset(file):
    df = read_jsonl_to_df(file)
    return df

def main():

    # Get configuration
    args = get_config()
    save_config(args, os.path.join(args.result_dir, 'config.json'))
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Training parameters
    learning_rate = args.learning_rate
    num_train_epochs = args.num_epochs
    #train_max_seq_len = 512
    batch_size = args.batch_size

    
    # Set up the fusion types
    if args.ensemble:
        FUSION_TYPES = ['concatenate', 'cross_attention', 'early', 'attention_weighted', 'transformer', 'late']
    else:
        FUSION_TYPES = [args.fusion]
    
    # Data files
    train_file = os.path.join(args.data_dir, 'arabic_hateful_meme_train.jsonl')
    validation_file = os.path.join(args.data_dir, 'arabic_hateful_meme_dev.jsonl')
    test_file = os.path.join(args.data_dir, 'arabic_hateful_meme_test.jsonl')
    
    
    # Prepare datasets
    train_df = prepare_dataset(train_file)
    val_df = prepare_dataset(validation_file)
    test_df = prepare_dataset(test_file)
    
    # Create datasets
    train_dataset = MultimodalDataset(train_df['id'], train_df['text'], train_df['img_path'], train_df['hate_label'], args)
    val_dataset = MultimodalDataset(val_df['id'], val_df['text'], val_df['img_path'], val_df['hate_label'], args)
    test_dataset = MultimodalDataset(test_df['id'], test_df['text'], test_df['img_path'], test_df['hate_label'], args)
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # Experiment with both loss types
    loss_types = ['weighted', 'focal']
    
    for loss_type in loss_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPERIMENT: {loss_type.upper()} LOSS")
        logger.info(f"{'='*80}")
        
        all_results = {}
        trained_models = {}
        
        for fusion_type in FUSION_TYPES:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training with {fusion_type.upper()} fusion - {loss_type.upper()} loss")
            logger.info(f"{'='*60}")
            
            # Initialize model
            model = MultimodalClassifier(num_classes=2, fusion_type=fusion_type, args=args)
            model.to(device)
            
            # Setup loss function
            if loss_type == 'weighted':
                test_hate_ratio = 213/2143
                test_nonhate_ratio = 1930/2143
                class_weights = torch.tensor([1/test_nonhate_ratio, 1/test_hate_ratio]).to(device)
                class_weights = class_weights / class_weights.sum() * 2
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:  # focal
                criterion = FocalLoss(alpha=0.75, gamma=2.0)
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, #weight_decay=1e-4
                                   )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
            
            best_val_macro_f1 = 0
            epoch_results = []
            
            # Training loop
            for epoch in range(num_train_epochs):
                train_loss, train_acc, train_f1, train_macro_f1 = train(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc, val_f1, val_macro_f1, _, _, _ = test(model, val_loader, criterion, device)
                
                epoch_results.append({
                    'epoch': epoch + 1,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'train_macro_f1': train_macro_f1,
                    'val_macro_f1': val_macro_f1,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
                
                logger.info(f'Epoch {epoch+1}/{num_train_epochs}:')
                logger.info(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Macro F1: {train_macro_f1:.4f}')
                logger.info(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Macro F1: {val_macro_f1:.4f}')
                
                scheduler.step(val_macro_f1)
                if val_macro_f1 > best_val_macro_f1:
                    best_val_macro_f1 = val_macro_f1
                    torch.save(model.state_dict(), os.path.join(args.model_dir, f'best_model_{fusion_type}_{loss_type}.pth'))
                    #torch.save(model.state_dict(), f'../models/best_model_{fusion_type}_{loss_type}.pth')
            
            # Save epoch results
            save_epoch_results(epoch_results, fusion_type, loss_type, 'training_results')
            
            # Load best model and test
            model.load_state_dict(model.state_dict(), os.path.join(args.model_dir, f'best_model_{fusion_type}_{loss_type}.pth'))
            #model.load_state_dict(torch.load(f'../models/best_model_{fusion_type}_{loss_type}.pth'))
            test_loss, test_acc, test_f1, test_macro_f1, test_preds, test_labels, test_probs = test(model, test_loader, criterion, device)
            
            all_results[fusion_type] = {
                'test_accuracy': test_acc,
                'test_f1': test_f1,
                'test_macro_f1': test_macro_f1,
                'test_loss': test_loss,
                'best_val_macro_f1': best_val_macro_f1
            }
            
            trained_models[fusion_type] = model
            
            logger.info(f'Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Macro F1: {test_macro_f1:.4f}')
            
            # Classification report
            report = classification_report(test_labels, test_preds, target_names=['not-hate', 'hate'])
            logger.info(f"Classification Report:\n{report}")
        
        # Ensemble predictions
        logger.info(f"\n{'='*60}")
        logger.info("ENSEMBLE PREDICTIONS")
        logger.info(f"{'='*60}")
        
        models_list = list(trained_models.values())
        fusion_types_list = list(trained_models.keys())
        ensemble = EnsembleClassifier(models_list, fusion_types_list, device)
        
        # Get true labels for ensemble evaluation
        true_labels = []
        with torch.no_grad():
            for data in test_loader:
                labels = data['hate_label']
                true_labels.extend(labels.numpy())
        
        # Majority vote
        majority_preds = ensemble.predict_majority_vote(test_loader)
        majority_acc = accuracy_score(true_labels, majority_preds)
        majority_f1 = f1_score(true_labels, majority_preds, average='weighted')
        majority_macro_f1 = f1_score(true_labels, majority_preds, average='macro')
        
        all_results['ensemble_majority'] = {
            'test_accuracy': majority_acc,
            'test_f1': majority_f1,
            'test_macro_f1': majority_macro_f1,
            'test_loss': 0.0,
            'best_val_macro_f1': 0.0
        }
        
        logger.info(f'Ensemble Majority Vote - Acc: {majority_acc:.4f}, F1: {majority_f1:.4f}, Macro F1: {majority_macro_f1:.4f}')
        
        # Equal weighted
        equal_weights = [1.0/len(models_list)] * len(models_list)
        equal_weighted_preds = ensemble.predict_weighted_vote(test_loader, equal_weights)
        equal_acc = accuracy_score(true_labels, equal_weighted_preds)
        equal_f1 = f1_score(true_labels, equal_weighted_preds, average='weighted')
        equal_macro_f1 = f1_score(true_labels, equal_weighted_preds, average='macro')
        
        all_results['ensemble_equal_weighted'] = {
            'test_accuracy': equal_acc,
            'test_f1': equal_f1,
            'test_macro_f1': equal_macro_f1,
            'test_loss': 0.0,
            'best_val_macro_f1': 0.0
        }
        
        logger.info(f'Ensemble Equal Weighted - Acc: {equal_acc:.4f}, F1: {equal_f1:.4f}, Macro F1: {equal_macro_f1:.4f}')
        
        # Transformer weighted
        transformer_weighted_preds = ensemble.predict_transformer_weighted(test_loader)
        trans_acc = accuracy_score(true_labels, transformer_weighted_preds)
        trans_f1 = f1_score(true_labels, transformer_weighted_preds, average='weighted')
        trans_macro_f1 = f1_score(true_labels, transformer_weighted_preds, average='macro')
        
        all_results['ensemble_transformer_weighted'] = {
            'test_accuracy': trans_acc,
            'test_f1': trans_f1,
            'test_macro_f1': trans_macro_f1,
            'test_loss': 0.0,
            'best_val_macro_f1': 0.0
        }
        
        logger.info(f'Ensemble Transformer Weighted - Acc: {trans_acc:.4f}, F1: {trans_f1:.4f}, Macro F1: {trans_macro_f1:.4f}')

if __name__ == "__main__":
    main()