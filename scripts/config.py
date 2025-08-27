import argparse
import json
import os

def get_config():
    parser = argparse.ArgumentParser(description='Multimodal Hate Speech Detection')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to directory containing train/dev/test jsonl files')
    
    # Model configuration
    parser.add_argument('--fusion', type=str, 
                       choices=['concatenate', 'cross_attention', 'early', 
                               'attention_weighted', 'transformer', 'late'],
                       default='transformer', help='Fusion method to use')
    
    parser.add_argument('--ensemble', type=bool, default=False,
                       help='Run ensemble of all fusion methods')
    
    # Model names
    parser.add_argument('--text_model', type=str, default='UBC-NLP/MARBERTv2',
                       help='Text encoder model name')
    parser.add_argument('--visual_model', type=str, default='openai/clip-vit-large-patch14',
                       help='Visual encoder model name')
    
    # Output directories
    parser.add_argument('--result_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--model_dir', type=str, default='./models',
                       help='Directory to save model checkpoints')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_seq_len', type=int, default=512)
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, 'training_logs'), exist_ok=True)
    
    return args

def save_config(args, filepath):
    """Save configuration to JSON file"""
    config_dict = vars(args)
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)