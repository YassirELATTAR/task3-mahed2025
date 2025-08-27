import os
import json
import pandas as pd
import numpy as np
import random
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2
import easyocr
from tqdm import tqdm
import shutil
from pathlib import Path
import uuid
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

class HateClassAugmenter:
    def __init__(self, source_data_path="Prop2Hate-Meme/", 
                 target_data_path="Prop2Hate-Meme-Aug/",
                 augmentation_factor=9):  # To balance classes (90.05% / 9.95% ≈ 9)
        
        self.source_data_path = source_data_path
        self.target_data_path = target_data_path
        self.augmentation_factor = augmentation_factor
        
        # Initialize OCR reader for Arabic text
        self.ocr_reader = easyocr.Reader(['ar', 'en'], gpu=True if torch.cuda.is_available() else False)
        
        # Create target directories
        self.setup_directories()
        
        # Define image augmentation techniques
        self.setup_image_augmentations()
        
    def setup_directories(self):
        """Create necessary directories for augmented data"""
        Path(self.target_data_path).mkdir(parents=True, exist_ok=True)
        Path(f"{self.target_data_path}/data").mkdir(parents=True, exist_ok=True)
        Path(f"{self.target_data_path}/augmented_images").mkdir(parents=True, exist_ok=True)
        
    def setup_image_augmentations(self):
        """Define various image augmentation techniques"""
        self.augmentation_pipelines = [
            # Pipeline 1: Rotation and scaling
            A.Compose([
                A.Rotate(limit=15, p=1.0),
                A.RandomScale(scale_limit=0.1, p=0.5),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
            ]),
            
            # Pipeline 2: Color and brightness
            A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.CLAHE(clip_limit=2.0, p=0.3),
            ]),
            
            # Pipeline 3: Noise and blur
            A.Compose([
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), p=0.5),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                ], p=0.7),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                ], p=0.5),
            ]),
            
            # Pipeline 4: Geometric transformations
            A.Compose([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.8),
                A.ElasticTransform(alpha=50, sigma=5, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
            ]),
            
            # Pipeline 5: Advanced augmentations
            A.Compose([
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, 
                              num_shadows_upper=2, shadow_dimension=5, p=0.3),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            ]),
            
            # Pipeline 6: Crop and pad variations
            A.Compose([
                A.RandomCrop(height=200, width=200, p=0.3),
                A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, 
                             value=0, p=1.0),
                A.Resize(224, 224, p=1.0),
            ]),
        ]
        
    def extract_text_with_ocr(self, image_path):
        """Extract text from image using OCR"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # Use EasyOCR to extract text
            results = self.ocr_reader.readtext(image)
            
            # Combine all detected text
            extracted_texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Only include text with reasonable confidence
                    extracted_texts.append(text.strip())
            
            if extracted_texts:
                combined_text = ' '.join(extracted_texts)
                # Clean the text
                combined_text = self.clean_extracted_text(combined_text)
                return combined_text
            
            return None
            
        except Exception as e:
            print(f"OCR extraction failed for {image_path}: {str(e)}")
            return None
    
    def clean_extracted_text(self, text):
        """Clean and normalize extracted text"""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters that might be OCR artifacts
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\w\.\,\!\?\:\;\-\(\)]', '', text)
        
        # Ensure minimum length
        if len(text.strip()) < 5:
            return None
            
        return text
    
    def augment_image(self, image_path, output_path, augmentation_id):
        """Apply image augmentation"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return False
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Select augmentation pipeline
            pipeline = self.augmentation_pipelines[augmentation_id % len(self.augmentation_pipelines)]
            
            # Apply augmentation
            augmented = pipeline(image=image)
            augmented_image = augmented['image']
            
            # Convert back to BGR and save
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, augmented_image)
            
            return True
            
        except Exception as e:
            print(f"Image augmentation failed for {image_path}: {str(e)}")
            return False
    
    def augment_sample(self, sample, augmentation_id):
        """Augment a single hate sample"""
        try:
            # Generate unique ID for augmented sample
            unique_id = str(uuid.uuid4())[:8]
            
            # Original image path
            original_img_path = sample['img_path'].replace('./data/', f'{self.source_data_path}/data/')
            
            if not os.path.exists(original_img_path):
                print(f"Original image not found: {original_img_path}")
                return None
            
            # Create augmented image path
            original_filename = os.path.basename(original_img_path)
            name, ext = os.path.splitext(original_filename)
            augmented_filename = f"{name}_aug_{unique_id}{ext}"
            augmented_img_path = f"{self.target_data_path}/augmented_images/{augmented_filename}"
            
            # Apply image augmentation
            if not self.augment_image(original_img_path, augmented_img_path, augmentation_id):
                return None
            
            # Extract text using OCR from augmented image
            extracted_text = self.extract_text_with_ocr(augmented_img_path)
            
            # Decide whether to use OCR text or original text
            use_ocr_text = (extracted_text is not None and 
                           len(extracted_text.strip()) >= 10 and 
                           random.random() < 0.7)  # 70% chance to use OCR if available
            
            final_text = extracted_text if use_ocr_text else sample['text']
            
            # Create augmented sample
            augmented_sample = {
                'id': f"augmented_{unique_id}_{sample['id']}",
                'text': final_text,
                'img_path': f"./data/augmented_images/{augmented_filename}",
                'hate_label': sample['hate_label'],
                'augmentation_type': f"pipeline_{augmentation_id}",
                'ocr_used': use_ocr_text,
                'original_id': sample['id']
            }
            
            return augmented_sample
            
        except Exception as e:
            print(f"Sample augmentation failed: {str(e)}")
            return None
    
    def augment_hate_samples(self, train_file):
        """Augment all hate samples from training data"""
        print(f"Loading training data from {train_file}")
        
        # Read training data
        train_data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line.strip()))
        
        # Filter hate samples
        hate_samples = [sample for sample in train_data if sample['hate_label'] == 1]
        non_hate_samples = [sample for sample in train_data if sample['hate_label'] == 0]
        
        print(f"Found {len(hate_samples)} hate samples and {len(non_hate_samples)} non-hate samples")
        print(f"Class distribution: {len(hate_samples)/(len(hate_samples)+len(non_hate_samples))*100:.2f}% hate")
        
        # Calculate how many augmented samples we need
        target_hate_count = len(non_hate_samples)  # Balance the classes
        needed_augmentations = max(0, target_hate_count - len(hate_samples))
        
        print(f"Need to generate {needed_augmentations} additional hate samples")
        
        augmented_samples = []
        
        # Generate augmented samples
        with tqdm(total=needed_augmentations, desc="Augmenting hate samples") as pbar:
            augmentation_count = 0
            
            while augmentation_count < needed_augmentations:
                for i, hate_sample in enumerate(hate_samples):
                    if augmentation_count >= needed_augmentations:
                        break
                    
                    # Create multiple augmentations per sample if needed
                    augmentation_id = augmentation_count % len(self.augmentation_pipelines)
                    
                    augmented_sample = self.augment_sample(hate_sample, augmentation_id)
                    
                    if augmented_sample is not None:
                        augmented_samples.append(augmented_sample)
                        augmentation_count += 1
                        pbar.update(1)
        
        print(f"Successfully generated {len(augmented_samples)} augmented hate samples")
        
        # Combine original and augmented data
        all_samples = train_data + augmented_samples
        
        # Shuffle the combined dataset
        random.shuffle(all_samples)
        
        # Save augmented training data
        output_file = f"{self.target_data_path}/arabic_hateful_meme_train_augmented.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Saved augmented training data to {output_file}")
        
        # Print final statistics
        final_hate_count = sum(1 for sample in all_samples if sample['hate_label'] == 1)
        final_non_hate_count = sum(1 for sample in all_samples if sample['hate_label'] == 0)
        
        print(f"\nFinal dataset statistics:")
        print(f"Total samples: {len(all_samples)}")
        print(f"Hate samples: {final_hate_count} ({final_hate_count/len(all_samples)*100:.2f}%)")
        print(f"Non-hate samples: {final_non_hate_count} ({final_non_hate_count/len(all_samples)*100:.2f}%)")
        print(f"Balance ratio: {final_hate_count/final_non_hate_count:.2f}")
        
        # Copy validation and test files (unchanged)
        self.copy_validation_test_files()
        
        return output_file
    
    def copy_validation_test_files(self):
        """Copy validation and test files to the new directory"""
        files_to_copy = [
            'arabic_hateful_meme_dev.jsonl',
            'arabic_hateful_meme_test.jsonl'
        ]
        
        for filename in files_to_copy:
            src = f"{self.source_data_path}/{filename}"
            dst = f"{self.target_data_path}/{filename}"
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"Copied {filename} to augmented directory")
            else:
                print(f"Warning: {src} not found")

def main():
    """Main function to run the augmentation process"""
    print("Starting Hate Class Data Augmentation...")
    
    # Initialize augmenter
    augmenter = HateClassAugmenter(
        source_data_path="Prop2Hate-Meme/",
        target_data_path="Prop2Hate-Meme-Aug/",
        augmentation_factor=9
    )
    
    # Run augmentation on training data
    train_file = "Prop2Hate-Meme/arabic_hateful_meme_train.jsonl"
    
    if not os.path.exists(train_file):
        print(f"Error: Training file {train_file} not found!")
        return
    
    augmented_train_file = augmenter.augment_hate_samples(train_file)
    
    print(f"\nAugmentation completed!")
    print(f"Augmented training data saved to: {augmented_train_file}")
    print(f"Use this new dataset path in your training: 'Prop2Hate-Meme-Aug/'")

if __name__ == "__main__":
    # Install required packages first
    print("Make sure you have installed the required packages:")
    print("pip install albumentations opencv-python easyocr")
    print()
    
    main()