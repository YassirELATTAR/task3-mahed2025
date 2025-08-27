import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer, CLIPProcessor
from torchvision import transforms

class MultimodalDataset(Dataset):
    def __init__(self, ids, text_data, image_data, labels, args, is_test=False):
        self.text_data = text_data
        self.image_data = image_data
        self.ids = ids
        self.is_test = is_test
        self.labels = labels
        self.data_dir = args.data_dir
        self.max_seq_len = args.max_seq_len
        
        # Initialize tokenizer and processor based on config
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        self.clip_processor = CLIPProcessor.from_pretrained(args.visual_model)
        
        # Fallback transform for CLIP compatibility
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                               (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        id = self.ids[index]
        text = self.text_data[index]
        
        # Fix image path relative to data_dir
        image_path = self.image_data[index]
        if not os.path.isabs(image_path):
            # Handle relative paths
            if image_path.startswith('./data/'):
                image_path = image_path.replace('./data/', '')
            image_path = os.path.join(self.data_dir, 'data', image_path)
        
        if not self.is_test:
            label = self.labels[index]

        text = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            max_length=self.max_seq_len, 
            padding='max_length',
            return_attention_mask=True, 
            return_tensors='pt'
        )

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