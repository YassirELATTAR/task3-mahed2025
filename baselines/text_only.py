import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn.functional as F
import numpy as np

# Create results directory
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('predictions', exist_ok=True)

# Training parameters
learning_rate = 1e-5
num_train_epochs = 20  # Reduced for baseline
train_max_seq_len = 512
batch_size = 16
weight_decay = 1e-4

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class TextOnlyDataset(Dataset):
    def __init__(self, ids, text_data, labels, is_test=False):
        self.text_data = text_data
        self.ids = ids
        self.is_test = is_test
        self.labels = labels
        
        # Use MARBERTv2 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('UBC-NLP/MARBERTv2')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        id = self.ids[index]
        text = self.text_data[index]
        
        if not self.is_test:
            label = self.labels[index]

        text = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            max_length=train_max_seq_len, 
            padding='max_length',
            return_attention_mask=True, 
            return_tensors='pt'
        )

        fdata = {
            'id': id,
            'text': text['input_ids'].squeeze(0),
            'text_mask': text['attention_mask'].squeeze(0),
        }

        if not self.is_test:
            fdata['hate_label'] = torch.tensor(label, dtype=torch.long)
            
        return fdata

class TextOnlyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(TextOnlyClassifier, self).__init__()
        
        # MARBERTv2 for text
        self.bert = AutoModel.from_pretrained('UBC-NLP/MARBERTv2')
        self.bert_drop = nn.Dropout(0.3)
        text_dim = self.bert.config.hidden_size  # 768
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, text, mask):
        # Text processing with MARBERTv2
        bert_output = self.bert(text, attention_mask=mask, return_dict=False)
        text_features = self.bert_drop(bert_output[0][:, 0, :])  # Use [CLS] token
        
        output = self.classifier(text_features)
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
        mask = data["text_mask"].to(device)
        labels = data['hate_label'].to(device)
        
        output = model(text, mask)
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

def test(model, test_loader, criterion, device, save_predictions=False, save_path=None):
    model.eval()
    test_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_ids = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            text = data["text"].to(device)
            mask = data["text_mask"].to(device)
            labels = data['hate_label'].to(device)
            ids = data['id']
            
            output = model(text, mask)
            loss = criterion(output, labels)
            
            probs = F.softmax(output, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(ids)
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Save predictions with probabilities
    if save_predictions and save_path:
        predictions_data = {
            'ids': all_ids,
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs,
            'confidence_scores': [max(prob) for prob in all_probs]
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(predictions_data, f)
        
        # Also save as CSV for readability
        csv_path = save_path.replace('.pkl', '.csv')
        df = pd.DataFrame({
            'id': all_ids,
            'prediction': all_preds,
            'true_label': all_labels,
            'confidence_score': [max(prob) for prob in all_probs],
            'prob_not_hate': [prob[0] for prob in all_probs],
            'prob_hate': [prob[1] for prob in all_probs]
        })
        df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {save_path} and {csv_path}")
    
    return test_loss, accuracy, f1, macro_f1, all_preds, all_labels, all_probs

def save_results_to_file(content, filename):
    """Simple function to save results to file"""
    with open(filename, 'a') as f:
        f.write(content + '\n')
    print(content)

def read_jsonl_to_df(filename):
    return pd.read_json(filename, lines=True)

def prepare_dataset(file):
    df = read_jsonl_to_df(file)
    return df

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Data files
    train_file = '../data/Prop2Hate-Meme/arabic_hateful_meme_train.jsonl'
    validation_file = '../data/Prop2Hate-Meme/arabic_hateful_meme_dev.jsonl' 
    test_file = '../data/Prop2Hate-Meme/arabic_hateful_meme_test.jsonl'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize experiment log
    log_file = 'text_only_experiment_log.txt'
    with open(log_file, 'w') as f:
        f.write("Text-Only Baseline Experiment Log\n")
        f.write("="*50 + "\n")
    
    save_results_to_file(f"Using device: {device}", log_file)
    
    # Prepare datasets
    train_df = prepare_dataset(train_file)
    val_df = prepare_dataset(validation_file)
    test_df = prepare_dataset(test_file)
    
    # Create datasets
    train_dataset = TextOnlyDataset(train_df['id'], train_df['text'], train_df['hate_label'])
    val_dataset = TextOnlyDataset(val_df['id'], val_df['text'], val_df['hate_label'])
    test_dataset = TextOnlyDataset(test_df['id'], test_df['text'], test_df['hate_label'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # Experiment with both loss types
    loss_types = ['weighted', 'focal']
    
    for loss_type in loss_types:
        save_results_to_file(f"\n{'='*80}", log_file)
        save_results_to_file(f"TEXT-ONLY BASELINE: {loss_type.upper()} LOSS", log_file)
        save_results_to_file(f"{'='*80}", log_file)
        
        # Initialize model
        model = TextOnlyClassifier(num_classes=2)
        model.to(device)
        
        # Setup loss function
        if loss_type == 'weighted':
            # Class weights inversely proportional to class frequencies
            hate_ratio = 0.0995
            nonhate_ratio = 0.9005
            class_weights = torch.tensor([1/nonhate_ratio, 1/hate_ratio]).to(device)
            class_weights = class_weights / class_weights.sum() * 2
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:  # focal
            criterion = FocalLoss(alpha=0.0995, gamma=2.0)
        
        # AdamW optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Linear scheduler
        total_steps = len(train_loader) * num_train_epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
        
        best_val_macro_f1 = 0
        patience_counter = 0
        
        # Training loop with early stopping
        for epoch in range(num_train_epochs):
            train_loss, train_acc, train_f1, train_macro_f1 = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, val_macro_f1, _, _, _ = test(model, val_loader, criterion, device)
            
            save_results_to_file(f'Epoch {epoch+1}/{num_train_epochs}:', log_file)
            save_results_to_file(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Macro F1: {train_macro_f1:.4f}', log_file)
            save_results_to_file(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Macro F1: {val_macro_f1:.4f}', log_file)
            
            scheduler.step()
            
            # Early stopping with patience 5
            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                patience_counter = 0
                torch.save(model.state_dict(), f'models/text_only_best_model_{loss_type}.pth')
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    save_results_to_file(f"Early stopping triggered at epoch {epoch+1}", log_file)
                    break
        
        # Load best model and test
        model.load_state_dict(torch.load(f'models/text_only_best_model_{loss_type}.pth'))
        
        # Test and save predictions
        predictions_path = f'predictions/text_only_predictions_{loss_type}.pkl'
        test_loss, test_acc, test_f1, test_macro_f1, test_preds, test_labels, test_probs = test(
            model, test_loader, criterion, device, 
            save_predictions=True, save_path=predictions_path
        )
        
        save_results_to_file(f'Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Macro F1: {test_macro_f1:.4f}', log_file)
        save_results_to_file(f'Best Val Macro F1: {best_val_macro_f1:.4f}', log_file)
        
        # Classification report
        report = classification_report(test_labels, test_preds, target_names=['not-hate', 'hate'])
        save_results_to_file(f"Classification Report:\n{report}", log_file)
        
        # Save model for later use in combination script
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_classes': 2,
                'loss_type': loss_type
            },
            'test_results': {
                'accuracy': test_acc,
                'f1': test_f1,
                'macro_f1': test_macro_f1,
                'best_val_macro_f1': best_val_macro_f1
            }
        }, f'models/text_only_complete_{loss_type}.pth')
        
        save_results_to_file(f"Model saved for combination use: text_only_complete_{loss_type}.pth", log_file)

def evaluate_gold(test_file="../data/task3_test_without_label.jsonl"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gold_df = prepare_dataset(test_file)
    # Ensure id is string on BOTH sides to avoid mismatched joins
    gold_df["id"] = gold_df["id"].astype(str)

    preds_df = pd.read_csv("../data/task3_test_gold.csv", dtype={"id": str})

    # figure out which column holds labels
    label_col = "testing_label" if "testing_label" in preds_df.columns else "prediction"
    if label_col not in preds_df.columns:
        raise ValueError(f"Couldn't find predictions column. Available columns: {list(preds_df.columns)}")

    # merge
    test_df = gold_df.merge(preds_df[["id", label_col]], on="id", how="left")

    # map to 0/1
    test_df["hate_label"] = test_df[label_col].map({"not-hate": 0, "hate": 1})

    # validate
    bad_mask = test_df["hate_label"].isna()
    if bad_mask.any():
        missing = test_df.loc[bad_mask, "id"].tolist()[:10]
        raise ValueError(
            f"{bad_mask.sum()} rows have missing/invalid labels after mapping. "
            f"Example ids: {missing}. "
            f"Make sure values are exactly 'hate' or 'not-hate' and all ids exist."
        )

    # cast to real integers
    test_df["hate_label"] = test_df["hate_label"].astype("int64")

    # keep a clean, positional index
    test_df = test_df.sort_values("id").reset_index(drop=True)

    # dataset/loader
    test_dataset = TextOnlyDataset(
        ids=test_df["id"],
        text_data=test_df["text"],
        labels=test_df["hate_label"],
        is_test=False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    log_file = 'gold_test_text_only_experiment_log.txt'
    loss_types = ['weighted', 'focal']
    for loss_type in loss_types:
        model = TextOnlyClassifier(num_classes=2)
        model.load_state_dict(torch.load(f'models/text_only_best_model_{loss_type}.pth'))
        model.to(device)

        if loss_type == 'weighted':
            hate_ratio = 0.0995
            nonhate_ratio = 0.9005
            class_weights = torch.tensor([1/nonhate_ratio, 1/hate_ratio], dtype=torch.float32).to(device)
            class_weights = class_weights / class_weights.sum() * 2
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = FocalLoss(alpha=0.0995, gamma=2.0)

        predictions_path = f'predictions/text_only_predictions_{loss_type}.pkl'
        test_loss, test_acc, test_f1, test_macro_f1, test_preds, test_labels, test_probs = test(
            model, test_loader, criterion, device, save_predictions=True, save_path=predictions_path
        )

        save_results_to_file(
            f'Gold Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, '
            f'F1: {test_f1:.4f}, Macro F1: {test_macro_f1:.4f}', log_file
        )
        report = classification_report(test_labels, test_preds, target_names=['not-hate', 'hate'])
        save_results_to_file(f"Gold Classification Report:\n{report}", log_file)

        


if __name__ == "__main__":

    # Just changes whether to run main() or evaluations of gold test split
    main()
    evaluate_gold()