from datasets import load_dataset
def download_prop2hate_meme():
    dataset = load_dataset("QCRI/Prop2Hate-Meme")

    # Specify the directory where you want to save the dataset

    output_dir="./Prop2Hate-Meme"

    # Save the dataset to the specified directory. This will save all splits to the output directory.
    dataset.save_to_disk(output_dir)

    # If you want to get the raw images from HF dataset format

    from PIL import Image
    import os
    import json

    # Directory to save the images
    output_dir="./Prop2Hate-Meme/"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the dataset and save each image
    for split in ['train','dev','test']:     
        jsonl_path = os.path.join(output_dir, f"arabic_hateful_meme_{split}.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(dataset[split]):
                # Access the image directly as it's already a PIL.Image object
                image = item['image']
                image_path = os.path.join(output_dir, item['img_path'])
                # Ensure the directory exists
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                image.save(image_path)
                del item['image']
                del item['prop_label']
                del item['hate_fine_grained_label']
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    download_prop2hate_meme()
