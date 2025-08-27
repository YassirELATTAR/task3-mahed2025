import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModel, CLIPProcessor
from torchvision import transforms
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

class ImageOnlyDataset(Dataset):
    def __init__(self, ids, image_data, labels, is_test=False):
        self.image_data = image_data
        self.ids = ids
        self.is_test = is_test
        self.labels = labels
        
        # Use CLIP processor for images
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
        
        # Fallback transform for CLIP compatibility
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        id = self.ids[index]
        
        image_path = self.image_data[index]
        if image_path.startswith('./data/'):
            image_path = image_path.replace('./data/', '../data/Prop2Hate-Meme/data/')
        
        if not self.is_test:
            label = self.labels[index]

        try:
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                raise FileNotFoundError
            
            image = Image.open(image_path).convert("RGB")
            image_inputs = self.clip_processor(images=image, return_tensors="pt")
            pixel_values = image_inputs['pixel_values'].squeeze(0)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            pixel_values = self.transform(image)
            print("Warning: Image processing with CLIP failed, using fallback transform.")

        fdata = {
            'id': id,
            'img_path': pixel_values,
        }

        if not self.is_test:
            fdata['hate_label'] = torch.tensor(label, dtype=torch.long)
            
        return fdata

class ImageOnlyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ImageOnlyClassifier, self).__init__()
        
        # CLIP Vision for images
        self.clip_vision = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')
        vision_dim = self.clip_vision.config.hidden_size  # 768
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(vision_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image):
        # Image processing with CLIP
        vision_output = self.clip_vision(pixel_values=image)
        vision_features = vision_output.pooler_output
        
        output = self.classifier(vision_features)
        return output

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []
    
    for data in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        image = data["img_path"].to(device)
        labels = data['hate_label'].to(device)
        
        output = model(image)
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
            image = data["img_path"].to(device)
            labels = data['hate_label'].to(device)
            ids = data['id']
            
            output = model(image)
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
    log_file = 'image_only_experiment_log.txt'
    with open(log_file, 'w') as f:
        f.write("Image-Only Baseline Experiment Log\n")
        f.write("="*50 + "\n")
    
    save_results_to_file(f"Using device: {device}", log_file)
    
    # Prepare datasets
    train_df = prepare_dataset(train_file)
    val_df = prepare_dataset(validation_file)
    test_df = prepare_dataset(test_file)
    
    # Create datasets
    train_dataset = ImageOnlyDataset(train_df['id'], train_df['img_path'], train_df['hate_label'])
    val_dataset = ImageOnlyDataset(val_df['id'], val_df['img_path'], val_df['hate_label'])
    test_dataset = ImageOnlyDataset(test_df['id'], test_df['img_path'], test_df['hate_label'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # Experiment with both loss types
    loss_types = ['weighted', 'focal']
    
    for loss_type in loss_types:
        save_results_to_file(f"\n{'='*80}", log_file)
        save_results_to_file(f"IMAGE-ONLY BASELINE: {loss_type.upper()} LOSS", log_file)
        save_results_to_file(f"{'='*80}", log_file)
        
        # Initialize model
        model = ImageOnlyClassifier(num_classes=2)
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
                torch.save(model.state_dict(), f'models/image_only_best_model_{loss_type}.pth')
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    save_results_to_file(f"Early stopping triggered at epoch {epoch+1}", log_file)
                    break
        
        # Load best model and test
        model.load_state_dict(torch.load(f'models/image_only_best_model_{loss_type}.pth'))
        
        # Test and save predictions
        predictions_path = f'predictions/image_only_predictions_{loss_type}.pkl'
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
        }, f'models/image_only_complete_{loss_type}.pth')
        
        save_results_to_file(f"Model saved for combination use: image_only_complete_{loss_type}.pth", log_file)

if __name__ == "__main__":
    main()