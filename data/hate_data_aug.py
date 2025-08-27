import os
import json
import random
from PIL import Image
import easyocr
from tweet_normalizer import ArabicTweetNormalizer

# Initialize EasyOCR reader for Arabic and English (English can help with mixed content)
# This will download models the first time it's run.
reader = easyocr.Reader(['ar', 'en'])

# Initialize the Arabic Tweet Normalizer
arabic_normalizer = ArabicTweetNormalizer()

def extract_text_from_image(image_path):
    try:
        # EasyOCR expects the path to the image
        result = reader.readtext(image_path)
        extracted_texts = [text for (bbox, text, prob) in result]
        return " ".join(extracted_texts)
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return ""

def augment_hate_data(input_file, output_file, image_base_path):
    augmented_data = []
    original_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            original_data.append(json.loads(line))

    hate_samples = [d for d in original_data if d['hate_label'] == 1]
    non_hate_count = len([d for d in original_data if d['hate_label'] == 0])

    print(f"Original Hate samples: {len(hate_samples)}")
    print(f"Original Non-Hate samples: {non_hate_count}")

    # Add all original data first
    augmented_data.extend(original_data)

    # Calculate how many hate samples we need to add to balance
    # Aim for roughly 1:1 ratio, or slightly less for hate if it becomes too dominant.
    # We want to make hate data "more", so let's aim for a count closer to non_hate_count.
    target_hate_count = non_hate_count
    num_to_add = target_hate_count - len(hate_samples)

    if num_to_add <= 0:
        print("Hate data is already sufficiently balanced or greater than non-hate data. No augmentation needed.")
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in original_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        return

    print(f"Adding {num_to_add} augmented hate samples.")

    # Duplicate and augment hate samples
    for i in range(num_to_add):
        if not hate_samples:
            break # No more hate samples to augment

        sample = random.choice(hate_samples).copy() # Deep copy to avoid modifying original

        # Fix the image path to match your own directory structure for OCR
        current_image_path = sample['img_path']
        if current_image_path.startswith('./data/'):
            current_image_path = current_image_path.replace('./data/', image_base_path)
        
        extracted_text = extract_text_from_image(current_image_path)
        # Check some random extracted texts from images:
        if i %300 == 0:
            print(f"Extracted text from {current_image_path}:\n*{extracted_text}*\n")
        
        # Randomly decide to add or replace text
        augmentation_strategy = random.choice(['duplicate_only', 'add_text', 'replace_text'])

        if augmentation_strategy == 'duplicate_only' or not extracted_text:
            # Just duplicate the sample as is, or if no text was extracted
            pass # No change to sample['text']
        elif augmentation_strategy == 'add_text':
            # Add extracted text to existing text
            sample['text'] = sample['text'] + " " + extracted_text
        elif augmentation_strategy == 'replace_text':
            # Replace existing text with extracted text
            sample['text'] = extracted_text

        # Apply Arabic Tweet Normalizer to the (potentially new) text
        sample['text'] = arabic_normalizer.normalize_tweet(sample['text'])
        
        # Assign a new unique ID (optional, but good practice if IDs are critical)
        # For simplicity here, we'll just append. If IDs need to be globally unique,
        # you might need a more sophisticated ID generation.
        # sample['id'] = f"{sample['id']}_aug_{_}" 
        
        augmented_data.append(sample)

    random.shuffle(augmented_data) # Shuffle the combined dataset

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in augmented_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print(f"Augmentation complete for {input_file}. New file saved to {output_file}")
    print(f"New total samples: {len(augmented_data)}")
    hate_count_after = len([d for d in augmented_data if d['hate_label'] == 1])
    print(f"New Hate samples: {hate_count_after}, Hate %: {hate_count_after / len(augmented_data) * 100:.2f}%")


if __name__ == "__main__":
    data_dir = 'Prop2Hate-Meme/'
    output_dir = 'Prop2Hate-Meme_augmented/'
    os.makedirs(output_dir, exist_ok=True)
    actual_image_base_path = '/Prop2Hate-Meme/data/data/' 
    image_root_for_ocr = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Prop2Hate-Meme', 'data'))
    print(f"Image root for OCR: {image_root_for_ocr}")


    augment_hate_data(os.path.join(data_dir, 'arabic_hateful_meme_train.jsonl'), 
                      os.path.join(output_dir, 'arabic_hateful_meme_train_augmented.jsonl'),
                      image_root_for_ocr)
    
    augment_hate_data(os.path.join(data_dir, 'arabic_hateful_meme_dev.jsonl'),
                      os.path.join(output_dir, 'arabic_hateful_meme_dev_augmented.jsonl'),
                      image_root_for_ocr)
    
    augment_hate_data(os.path.join(data_dir, 'arabic_hateful_meme_test.jsonl'),
                      os.path.join(output_dir, 'arabic_hateful_meme_test_augmented.jsonl'),
                      image_root_for_ocr)

    print(f"Augmented datasets saved to {output_dir}")
    print("Remember to update your main training script to use these new augmented files.")