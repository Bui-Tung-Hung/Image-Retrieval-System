#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Vietnamese Flickr8k Dataset class for BEiT3
"""

import os
import json
import torch
from datasets import BaseDataset

class VietnameseFlickr8kDataset(BaseDataset):
    """
    Vietnamese Flickr8k dataset for image captioning
    Follows the same pattern as CaptioningDataset in BEiT3
    """
    
    def __init__(self, data_path, split, transform, 
                tokenizer, num_max_bpe_tokens, task, mask_prob=0.6):
        super().__init__(
            data_path=data_path, split=split, 
            transform=transform, tokenizer=tokenizer, 
            num_max_bpe_tokens=num_max_bpe_tokens, task=task, 
        )
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = mask_prob

    @staticmethod
    def get_index_files(split, task=None):
        """Define index files for Vietnamese Flickr8k dataset"""
        if split == "train":
            return ("vietnamese_flickr8k.train.jsonl", )
        elif split == "val":
            return ("vietnamese_flickr8k.val.jsonl", )
        elif split == "test":
            return ("vietnamese_flickr8k.test.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def _get_mask_token(self, token):
        """Masking strategy for training"""
        import random
        p = random.random()
        if p < 0.8:
            return self.mask_token_id
        elif p < 0.9:
            return token
        else:
            return random.randint(3, self.language_vocab_size - 1)

    def _masking_on_text_tokens(self, tokens, num_tokens, mask_prob):
        """Apply masking to text tokens for training"""
        import random
        bool_masked_pos = [0] * len(tokens)
        to_mask = min(int(num_tokens * mask_prob + 0.5), num_tokens - 1)
        to_mask = max(to_mask, 1)
        num_masked_tokens = 0
        while num_masked_tokens < to_mask:
            i = random.randint(1, num_tokens - 1)
            if bool_masked_pos[i] == 0:
                bool_masked_pos[i] = 1
                tokens[i] = self._get_mask_token(tokens[i])
                num_masked_tokens += 1

        return tokens, bool_masked_pos

    def __getitem__(self, index: int):
        """Get a single training/validation example"""
        data = dict()
        item = self.items[index]
        
        # Load image
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["image_id"] = item.get("image_id", index)

        # Process text
        text_segment = item["text_segment"]
        if text_segment is not None:
            language_tokens, padding_mask, num_tokens = self._get_text_segment(text_segment)
            
            # Apply masking during training
            if self.split == "train":
                masked_tokens = language_tokens[:]
                masked_tokens, language_masked_pos = \
                    self._masking_on_text_tokens(masked_tokens, num_tokens, self.mask_prob)
                data["masked_tokens"] = masked_tokens
                data["language_masked_pos"] = language_masked_pos
            
            data["language_tokens"] = language_tokens
            data["padding_mask"] = padding_mask
            
        return data

    @staticmethod
    def create_vietnamese_flickr8k_index(
        data_path, 
        tokenizer, 
        vietnamese_captions_file,
        images_dir="images"
    ):
        """
        Create dataset index files for Vietnamese Flickr8k
        
        Args:
            data_path: Directory to save index files
            tokenizer: BEiT3 tokenizer
            vietnamese_captions_file: Path to Vietnamese captions file
            images_dir: Directory containing images
        """
        
        print("Creating Vietnamese Flickr8k dataset indices...")
        
        # Load Vietnamese captions
        # Expected format: image_filename.jpg\tVietnamese caption
        image_to_captions = {}
        
        if os.path.exists(vietnamese_captions_file):
            with open(vietnamese_captions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        image_file = parts[0]
                        caption = parts[1]
                        
                        if image_file not in image_to_captions:
                            image_to_captions[image_file] = []
                        image_to_captions[image_file].append(caption)
        else:
            print(f"Warning: {vietnamese_captions_file} not found. Using sample data.")
            # Use sample data for demonstration
            image_to_captions = {
                "example1.jpg": [
                    "Một con chó nhỏ đang chạy trên bãi cỏ xanh",
                    "Con chó con màu nâu đang vui vẻ chạy nhảy"
                ],
                "example2.jpg": [
                    "Có một người phụ nữ đang ngồi trên ghế đá",
                    "Cô gái mặc áo đỏ đang nghỉ ngơi trong công viên"
                ],
                "example3.jpg": [
                    "Trẻ em đang chơi bóng trong công viên",
                    "Các em nhỏ vui vẻ chơi đùa ngoài trời"
                ]
            }
        
        # Split data (80% train, 10% val, 10% test)
        all_images = list(image_to_captions.keys())
        train_split = int(0.8 * len(all_images))
        val_split = int(0.9 * len(all_images))
        
        splits = {
            "train": all_images[:train_split],
            "val": all_images[train_split:val_split],
            "test": all_images[val_split:]
        }
        
        for split_name, image_list in splits.items():
            items = []
            image_counter = set()
            
            for image_file in image_list:
                image_path = os.path.join(images_dir, image_file)
                captions = image_to_captions[image_file]
                
                for caption in captions:
                    # Tokenize Vietnamese caption
                    tokens = tokenizer.tokenize(caption)
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    
                    items.append({
                        "image_path": image_path,
                        "text_segment": token_ids,
                        "image_id": len(image_counter),
                    })
                
                if image_path not in image_counter:
                    image_counter.add(image_path)
            
            # Save to JSONL
            index_file = os.path.join(data_path, f"vietnamese_flickr8k.{split_name}.jsonl")
            with open(index_file, mode="w", encoding="utf-8") as writer:
                for data in items:
                    writer.write(json.dumps(data, ensure_ascii=False))
                    writer.write('\n')
            
            print(f"✓ {split_name}: {len(image_counter)} images, {len(items)} pairs -> {index_file}")

def test_vietnamese_dataset():
    """Test the Vietnamese dataset implementation"""
    from transformers import XLMRobertaTokenizer
    from torchvision import transforms
    from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    
    print("=== Testing Vietnamese Flickr8k Dataset ===\n")
    
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer("beit3.spm")
    
    # Create dataset indices
    VietnameseFlickr8kDataset.create_vietnamese_flickr8k_index(
        data_path=".",
        tokenizer=tokenizer,
        vietnamese_captions_file="vietnamese_captions.txt"  # Your actual file
    )
    
    # Create transforms (same as BEiT3)
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3), 
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])
    
    # Test dataset loading
    try:
        dataset = VietnameseFlickr8kDataset(
            data_path=".",
            split="train",
            transform=transform,
            tokenizer=tokenizer,
            num_max_bpe_tokens=64,
            task="vietnamese_flickr8k",
            mask_prob=0.6
        )
        
        print(f"✓ Dataset created successfully with {len(dataset)} items")
        
        # Test getting an item (without actual image files)
        print("\n=== Dataset Structure ===")
        print(f"Index files: {dataset.get_index_files('train')}")
        print(f"Items loaded: {len(dataset.items)}")
        
        if len(dataset.items) > 0:
            sample_item = dataset.items[0]
            print(f"Sample item keys: {sample_item.keys()}")
            
            decoded = tokenizer.decode(sample_item['text_segment'], skip_special_tokens=True)
            print(f"Sample caption: {decoded}")
        
        print("\n✓ Vietnamese Flickr8k dataset is ready for BEiT3 fine-tuning!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_vietnamese_dataset()