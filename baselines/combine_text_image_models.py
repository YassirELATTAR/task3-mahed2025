import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, CLIPVisionModel, CLIPProcessor
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from visual_only import ImageOnlyClassifier
from text_only import TextOnlyClassifier


class CombinedDataset(Dataset):
    def __init__(self, ids, text_data, image_data, labels, is_test=False):
        self.text_data = text_data
        self.image_data = image_data
        self.ids = ids
        self.is_test = is_test
        self.labels = labels
        
        # Use MARBERTv2 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('UBC-NLP/MARBERTv2')
        
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
        text = self.text_data[index]
        
        image_path = self.image_data[index]
        if image_path.startswith('./data/'):
            image_path = image_path.replace('./data/', '../data/Prop2Hate-Meme/data/')
        
        if not self.is_test:
            label = self.labels[index]

        # Process text
        text = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            max_length=512, 
            padding='max_length',
            return_attention_mask=True, 
            return_tensors='pt'
        )

        # Process image
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
            'text': text['input_ids'].squeeze(0),
            'text_mask': text['attention_mask'].squeeze(0),
            'img_path': pixel_values,
        }

        if not self.is_test:
            fdata['hate_label'] = torch.tensor(label, dtype=torch.long)
            
        return fdata

class SeparateBaselineCombiner:
    def __init__(self, text_model, image_model, device):
        self.text_model = text_model
        self.image_model = image_model
        self.device = device
        
    def predict_with_confidence_voting(self, test_loader, confidence_threshold=0.7):
        """
        Combine predictions using confidence-based voting:
        - If both models agree, use that prediction
        - If they disagree, use the prediction from the model with higher confidence
        - If both have low confidence, use a weighted average
        """
        self.text_model.eval()
        self.image_model.eval()
        
        all_preds = []
        all_labels = []
        all_ids = []
        all_text_probs = []
        all_image_probs = []
        all_combined_probs = []
        decision_types = []  # Track how each decision was made
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Combined Prediction"):
                text = data["text"].to(self.device)
                image = data["img_path"].to(self.device)
                mask = data["text_mask"].to(self.device)
                labels = data['hate_label'].to(self.device)
                ids = data['id']
                
                # Get predictions from both models
                text_output = self.text_model(text, mask)
                image_output = self.image_model(image)
                
                text_probs = F.softmax(text_output, dim=1)
                image_probs = F.softmax(image_output, dim=1)
                
                # Get predictions and confidence scores
                _, text_preds = torch.max(text_probs, 1)
                _, image_preds = torch.max(image_probs, 1)
                
                text_confidence = torch.max(text_probs, 1)[0]
                image_confidence = torch.max(image_probs, 1)[0]
                
                # Decision logic for each sample in the batch
                batch_size = text_probs.size(0)
                combined_preds = []
                combined_probs = []
                batch_decision_types = []
                
                for i in range(batch_size):
                    text_pred = text_preds[i].item()
                    image_pred = image_preds[i].item()
                    text_conf = text_confidence[i].item()
                    image_conf = image_confidence[i].item()
                    
                    if text_pred == image_pred:
                        # Both models agree
                        final_pred = text_pred
                        final_prob = (text_probs[i] + image_probs[i]) / 2
                        decision_type = 'agreement'
                    elif text_conf > confidence_threshold and image_conf <= confidence_threshold:
                        # Only text model is confident
                        final_pred = text_pred
                        final_prob = text_probs[i]
                        decision_type = 'text_confident'
                    elif image_conf > confidence_threshold and text_conf <= confidence_threshold:
                        # Only image model is confident
                        final_pred = image_pred
                        final_prob = image_probs[i]
                        decision_type = 'image_confident'
                    elif text_conf > image_conf:
                        # Text model is more confident
                        final_pred = text_pred
                        final_prob = text_probs[i]
                        decision_type = 'text_higher_conf'
                    else:
                        # Image model is more confident
                        final_pred = image_pred
                        final_prob = image_probs[i]
                        decision_type = 'image_higher_conf'
                    
                    combined_preds.append(final_pred)
                    combined_probs.append(final_prob.cpu().numpy())
                    batch_decision_types.append(decision_type)
                
                all_preds.extend(combined_preds)
                all_labels.extend(labels.cpu().numpy())
                all_ids.extend(ids)
                all_text_probs.extend(text_probs.cpu().numpy())
                all_image_probs.extend(image_probs.cpu().numpy())
                all_combined_probs.extend(combined_probs)
                decision_types.extend(batch_decision_types)
        
        return {
            'predictions': all_preds,
            'true_labels': all_labels,
            'ids': all_ids,
            'text_probabilities': all_text_probs,
            'image_probabilities': all_image_probs,
            'combined_probabilities': all_combined_probs,
            'decision_types': decision_types
        }
    
    def predict_simple_average(self, test_loader):
        """Simple average of probabilities from both models"""
        self.text_model.eval()
        self.image_model.eval()
        
        all_preds = []
        all_labels = []
        all_ids = []
        all_combined_probs = []
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Simple Average Prediction"):
                text = data["text"].to(self.device)
                image = data["img_path"].to(self.device)
                mask = data["text_mask"].to(self.device)
                labels = data['hate_label'].to(self.device)
                ids = data['id']
                
                # Get predictions from both models
                text_output = self.text_model(text, mask)
                image_output = self.image_model(image)
                
                text_probs = F.softmax(text_output, dim=1)
                image_probs = F.softmax(image_output, dim=1)
                
                # Simple average
                combined_probs = (text_probs + image_probs) / 2
                _, combined_preds = torch.max(combined_probs, 1)
                
                all_preds.extend(combined_preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_ids.extend(ids)
                all_combined_probs.extend(combined_probs.cpu().numpy())
        
        return all_preds, all_labels, all_ids, all_combined_probs
    
    def predict_weighted_average(self, test_loader, text_weight=0.5, image_weight=0.5):
        """Weighted average of probabilities from both models"""
        self.text_model.eval()
        self.image_model.eval()
        
        all_preds = []
        all_labels = []
        all_ids = []
        all_combined_probs = []
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Weighted Average Prediction"):
                text = data["text"].to(self.device)
                image = data["img_path"].to(self.device)
                mask = data["text_mask"].to(self.device)
                labels = data['hate_label'].to(self.device)
                ids = data['id']
                
                # Get predictions from both models
                text_output = self.text_model(text, mask)
                image_output = self.image_model(image)
                
                text_probs = F.softmax(text_output, dim=1)
                image_probs = F.softmax(image_output, dim=1)
                
                # Weighted average
                combined_probs = text_weight * text_probs + image_weight * image_probs
                _, combined_preds = torch.max(combined_probs, 1)
                
                all_preds.extend(combined_preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_ids.extend(ids)
                all_combined_probs.extend(combined_probs.cpu().numpy())
        
        return all_preds, all_labels, all_ids, all_combined_probs

def save_results_to_file(content, filename):
    """Simple function to save results to file"""
    with open(filename, 'a') as f:
        f.write(content + '\n')
    print(content)

def load_model(model_path, model_class, device):
    """Load a saved model"""
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class(num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model, checkpoint

def read_jsonl_to_df(filename):
    return pd.read_json(filename, lines=True)

def prepare_dataset(file):
    df = read_jsonl_to_df(file)
    return df

def analyze_decision_types(decision_types, results):
    """Analyze how different decision types perform"""
    decision_analysis = {}
    for decision_type in set(decision_types):
        indices = [i for i, dt in enumerate(decision_types) if dt == decision_type]
        if indices:
            preds = [results['predictions'][i] for i in indices]
            labels = [results['true_labels'][i] for i in indices]
            
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')
            count = len(indices)
            
            decision_analysis[decision_type] = {
                'count': count,
                'accuracy': accuracy,
                'macro_f1': f1,
                'percentage': count / len(decision_types) * 100
            }
    
    return decision_analysis

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize experiment log
    log_file = 'separate_baseline_experiment_log.txt'
    with open(log_file, 'w') as f:
        f.write("Separate Baseline Combination Experiment Log\n")
        f.write("="*50 + "\n")
    
    save_results_to_file(f"Using device: {device}", log_file)
    
    # Data files
    test_file = '../data/Prop2Hate-Meme/arabic_hateful_meme_test.jsonl'
    test_df = prepare_dataset(test_file)
    
    # Create combined dataset for testing
    test_dataset = CombinedDataset(test_df['id'], test_df['text'], test_df['img_path'], test_df['hate_label'])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False)
    
    # Test different loss types
    loss_types = ['weighted', 'focal']
    
    for loss_type in loss_types:
        save_results_to_file(f"\n{'='*80}", log_file)
        save_results_to_file(f"SEPARATE BASELINE COMBINATION: {loss_type.upper()} LOSS", log_file)
        save_results_to_file(f"{'='*80}", log_file)
        
        # Load pre-trained models
        text_model_path = f'models/text_only_complete_{loss_type}.pth'
        image_model_path = f'models/image_only_complete_{loss_type}.pth'
        
        if not os.path.exists(text_model_path) or not os.path.exists(image_model_path):
            save_results_to_file(f"Model files not found for {loss_type} loss. Please run text and image baseline scripts first.", log_file)
            continue
        
        text_model, text_checkpoint = load_model(text_model_path, TextOnlyClassifier, device)
        image_model, image_checkpoint = load_model(image_model_path, ImageOnlyClassifier, device)
        
        save_results_to_file(f"Loaded text model - Test Macro F1: {text_checkpoint['test_results']['macro_f1']:.4f}", log_file)
        save_results_to_file(f"Loaded image model - Test Macro F1: {image_checkpoint['test_results']['macro_f1']:.4f}", log_file)
        
        # Initialize combiner
        combiner = SeparateBaselineCombiner(text_model, image_model, device)
        
        # Method 1: Confidence-based voting
        save_results_to_file("\n--- Confidence-Based Voting ---", log_file)
        confidence_results = combiner.predict_with_confidence_voting(test_loader, confidence_threshold=0.7)
        
        conf_accuracy = accuracy_score(confidence_results['true_labels'], confidence_results['predictions'])
        conf_f1 = f1_score(confidence_results['true_labels'], confidence_results['predictions'], average='weighted')
        conf_macro_f1 = f1_score(confidence_results['true_labels'], confidence_results['predictions'], average='macro')
        
        save_results_to_file(f"Confidence-based - Acc: {conf_accuracy:.4f}, F1: {conf_f1:.4f}, Macro F1: {conf_macro_f1:.4f}", log_file)
        
        # Analyze decision types
        decision_analysis = analyze_decision_types(confidence_results['decision_types'], confidence_results)
        save_results_to_file("\nDecision Type Analysis:", log_file)
        for decision_type, stats in decision_analysis.items():
            save_results_to_file(f"  {decision_type}: {stats['count']} samples ({stats['percentage']:.1f}%) - "
                               f"Acc: {stats['accuracy']:.4f}, Macro F1: {stats['macro_f1']:.4f}", log_file)
        
        # Save confidence-based predictions
        conf_pred_path = f'predictions/separate_confidence_predictions_{loss_type}.pkl'
        with open(conf_pred_path, 'wb') as f:
            pickle.dump(confidence_results, f)
        
        # Method 2: Simple average
        save_results_to_file("\n--- Simple Average ---", log_file)
        simple_preds, simple_labels, simple_ids, simple_probs = combiner.predict_simple_average(test_loader)
        
        simple_accuracy = accuracy_score(simple_labels, simple_preds)
        simple_f1 = f1_score(simple_labels, simple_preds, average='weighted')
        simple_macro_f1 = f1_score(simple_labels, simple_preds, average='macro')
        
        save_results_to_file(f"Simple Average - Acc: {simple_accuracy:.4f}, F1: {simple_f1:.4f}, Macro F1: {simple_macro_f1:.4f}", log_file)
        
        # Method 3: Weighted average (based on individual model performance)
        save_results_to_file("\n--- Weighted Average ---", log_file)
        text_macro_f1 = text_checkpoint['test_results']['macro_f1']
        image_macro_f1 = image_checkpoint['test_results']['macro_f1']
        
        # Weight based on individual performance
        total_performance = text_macro_f1 + image_macro_f1
        text_weight = text_macro_f1 / total_performance
        image_weight = image_macro_f1 / total_performance
        
        save_results_to_file(f"Using weights - Text: {text_weight:.3f}, Image: {image_weight:.3f}", log_file)
        
        weighted_preds, weighted_labels, weighted_ids, weighted_probs = combiner.predict_weighted_average(
            test_loader, text_weight, image_weight)
        
        weighted_accuracy = accuracy_score(weighted_labels, weighted_preds)
        weighted_f1 = f1_score(weighted_labels, weighted_preds, average='weighted')
        weighted_macro_f1 = f1_score(weighted_labels, weighted_preds, average='macro')
        
        save_results_to_file(f"Weighted Average - Acc: {weighted_accuracy:.4f}, F1: {weighted_f1:.4f}, Macro F1: {weighted_macro_f1:.4f}", log_file)
        
        # Save all results
        combined_results = {
            'loss_type': loss_type,
            'confidence_based': {
                'predictions': confidence_results['predictions'],
                'accuracy': conf_accuracy,
                'f1': conf_f1,
                'macro_f1': conf_macro_f1,
                'decision_analysis': decision_analysis
            },
            'simple_average': {
                'predictions': simple_preds,
                'accuracy': simple_accuracy,
                'f1': simple_f1,
                'macro_f1': simple_macro_f1
            },
            'weighted_average': {
                'predictions': weighted_preds,
                'accuracy': weighted_accuracy,
                'f1': weighted_f1,
                'macro_f1': weighted_macro_f1,
                'text_weight': text_weight,
                'image_weight': image_weight
            },
            'true_labels': simple_labels,  # Same for all methods
            'ids': simple_ids
        }
        
        # Save comprehensive results
        results_path = f'predictions/separate_baseline_results_{loss_type}.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(combined_results, f)
        
        # Classification reports
        save_results_to_file("\n--- Classification Reports ---", log_file)
        
        save_results_to_file("\nConfidence-Based Classification Report:", log_file)
        conf_report = classification_report(confidence_results['true_labels'], confidence_results['predictions'], 
                                          target_names=['not-hate', 'hate'])
        save_results_to_file(conf_report, log_file)
        
        save_results_to_file("\nSimple Average Classification Report:", log_file)
        simple_report = classification_report(simple_labels, simple_preds, target_names=['not-hate', 'hate'])
        save_results_to_file(simple_report, log_file)
        
        save_results_to_file("\nWeighted Average Classification Report:", log_file)
        weighted_report = classification_report(weighted_labels, weighted_preds, target_names=['not-hate', 'hate'])
        save_results_to_file(weighted_report, log_file)
        
        # Summary
        save_results_to_file(f"\n--- SUMMARY for {loss_type.upper()} LOSS ---", log_file)
        save_results_to_file(f"Text-only baseline: Macro F1 = {text_macro_f1:.4f}", log_file)
        save_results_to_file(f"Image-only baseline: Macro F1 = {image_macro_f1:.4f}", log_file)
        save_results_to_file(f"Confidence-based combination: Macro F1 = {conf_macro_f1:.4f}", log_file)
        save_results_to_file(f"Simple average combination: Macro F1 = {simple_macro_f1:.4f}", log_file)
        save_results_to_file(f"Weighted average combination: Macro F1 = {weighted_macro_f1:.4f}", log_file)
        
        best_method = max([
            ('Text-only', text_macro_f1),
            ('Image-only', image_macro_f1),
            ('Confidence-based', conf_macro_f1),
            ('Simple average', simple_macro_f1),
            ('Weighted average', weighted_macro_f1)
        ], key=lambda x: x[1])
        
        save_results_to_file(f"\nBest method: {best_method[0]} with Macro F1 = {best_method[1]:.4f}", log_file)

if __name__ == "__main__":
    main()