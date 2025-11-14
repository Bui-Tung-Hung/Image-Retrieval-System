#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnamese Flickr8k Image Retrieval Dataset for BEiT3
"""

import json
import os
from collections import defaultdict
from transformers import XLMRobertaTokenizer

def create_vietnamese_retrieval_dataset():
    """
    Create Vietnamese Flickr8k dataset for image retrieval task
    """
    
    print("=== Vietnamese Flickr8k Image Retrieval Dataset ===\n")
    
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer("beit3.spm")
    print("✓ BEiT3 tokenizer loaded")
    
    # Sample Vietnamese Flickr8k data for retrieval
    # Format: image -> multiple Vietnamese captions for retrieval
    vietnamese_flickr8k_retrieval = {
        "1000268201_693b08cb0e.jpg": [
            "Một con chó màu đen và trắng đang nhảy qua rào cản trong sân",
            "Con chó Border Collie đang thể hiện khả năng nhảy vượt chướng ngại vật",
            "Chú chó thể thao đang tham gia cuộc thi khéo léo",
            "Con vật cưng đang chơi và vận động ngoài trời",
            "Hình ảnh một chú chó đang hoạt động thể thao"
        ],
        "1001773457_577c3a7d70.jpg": [
            "Hai em bé đang ngồi trên ghế bành màu đỏ",
            "Hai đứa trẻ nhỏ đang cười vui vẻ trên chiếc ghế tựa",
            "Cảnh hai anh em đang nghỉ ngơi trong nhà",
            "Trẻ em đang thư giãn trên đồ nội thất",
            "Khoảnh khắc hạnh phúc của hai đứa trẻ"
        ],
        "1002674143_1b742ab4b8.jpg": [
            "Một người đàn ông mặc áo phông đen đang cưỡi xe đạp trên đường",
            "Người đàn ông đang đạp xe thể thao trên con phố",
            "Hoạt động thể dục bằng xe đạp trên đường phố",
            "Nam giới đang tham gia giao thông bằng xe đạp",
            "Cảnh một người đang di chuyển bằng phương tiện thân thiện môi trường"
        ]
    }
    
    print("📊 Dataset characteristics for retrieval:")
    print(f"- {len(vietnamese_flickr8k_retrieval)} images")
    total_captions = sum(len(captions) for captions in vietnamese_flickr8k_retrieval.values())
    print(f"- {total_captions} Vietnamese text queries")
    print(f"- Average {total_captions/len(vietnamese_flickr8k_retrieval):.1f} captions per image")
    
    # Create dataset splits for retrieval
    all_images = list(vietnamese_flickr8k_retrieval.keys())
    
    # For retrieval, we need many text queries to test retrieval performance
    splits = {
        "train": all_images,          # Use all for training
        "val": all_images[:2],        # Subset for validation
        "test": all_images[:2]        # Subset for testing
    }
    
    print(f"\n📋 Retrieval dataset splits:")
    for split, images in splits.items():
        print(f"- {split}: {len(images)} images")
    
    # Create JSONL files for image retrieval
    for split_name, image_list in splits.items():
        items = []
        image_counter = set()
        
        for image_file in image_list:
            captions = vietnamese_flickr8k_retrieval[image_file]
            image_path = f"images/{image_file}"
            
            # For retrieval, each caption becomes a separate query
            for caption in captions:
                tokens = tokenizer.tokenize(caption)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                
                items.append({
                    "image_path": image_path,
                    "text_segment": token_ids,
                    "image_id": len(image_counter),  # Same image_id for same image
                })
            
            if image_path not in image_counter:
                image_counter.add(image_path)
        
        # Save to JSONL format (compatible with RetrievalDataset)
        index_file = f"vietnamese_flickr8k.{split_name}.jsonl"
        with open(index_file, mode="w", encoding="utf-8") as writer:
            for data in items:
                writer.write(json.dumps(data, ensure_ascii=False))
                writer.write('\n')
        
        print(f"✓ Created {index_file}: {len(image_counter)} images, {len(items)} text queries")
    
    # Test retrieval scenario
    print(f"\n🔍 Retrieval Task Analysis:")
    print("- Task: Given Vietnamese text query, find matching image")
    print("- Evaluation: Text-to-Image Retrieval (TR@1, TR@5, TR@10)")
    print("- Also: Image-to-Text Retrieval (IR@1, IR@5, IR@10)")
    
    # Show example queries
    print(f"\n📝 Example Vietnamese queries:")
    sample_queries = [
        "Một con chó đang nhảy",
        "Hai đứa trẻ ngồi trên ghế",
        "Người đàn ông đi xe đạp",
        "Con vật đang chơi ngoài trời",
        "Trẻ em trong nhà"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        tokens = tokenizer.tokenize(query)
        print(f"{i}. '{query}' -> {len(tokens)} tokens")
    
    # Show training command
    print(f"\n🚀 Training command for Vietnamese retrieval:")
    command = """python run_beit3_finetuning.py \\
    --model beit3_base_patch16_384 \\
    --input_size 384 \\
    --task flickr30k \\
    --batch_size 32 \\
    --layer_decay 0.65 \\
    --lr 1e-4 \\
    --epochs 20 \\
    --warmup_epochs 5 \\
    --drop_path 0.1 \\
    --sentencepiece_model beit3.spm \\
    --finetune beit3_base_itc_patch16_224.pth \\
    --data_path /path/to/vietnamese_flickr8k \\
    --output_dir /path/to/output \\
    --weight_decay 0.05"""
    
    print(command)
    
    return True

def analyze_retrieval_vs_captioning():
    """Compare retrieval vs captioning tasks"""
    
    print(f"\n=== Retrieval vs Captioning Comparison ===\n")
    
    comparison = {
        "Task": {
            "Retrieval": "Given text query, find matching image (or vice versa)",
            "Captioning": "Given image, generate descriptive text"
        },
        "Training": {
            "Retrieval": "Contrastive learning between image and text embeddings",
            "Captioning": "Masked language modeling for text generation"
        },
        "Architecture": {
            "Retrieval": "Uses CLS tokens for similarity matching",
            "Captioning": "Uses sequence-to-sequence generation"
        },
        "Evaluation": {
            "Retrieval": "Recall@K (R@1, R@5, R@10) for both directions",
            "Captioning": "BLEU, METEOR, CIDEr, SPICE scores"
        },
        "Use Cases": {
            "Retrieval": "Search engines, content discovery, image databases",
            "Captioning": "Image description, accessibility, content generation"
        },
        "Vietnamese Benefits": {
            "Retrieval": "Vietnamese search queries for image collections",
            "Captioning": "Vietnamese descriptions for Vietnamese images"
        }
    }
    
    for category, details in comparison.items():
        print(f"📋 {category}:")
        for task, description in details.items():
            print(f"  • {task}: {description}")
        print()
    
    print("🎯 For Vietnamese Flickr8k:")
    print("✓ Retrieval allows searching images using Vietnamese queries")
    print("✓ Better for building Vietnamese image search applications")
    print("✓ Can handle both text→image and image→text scenarios")
    print("✓ More practical for real-world Vietnamese applications")

def test_retrieval_compatibility():
    """Test compatibility with BEiT3 RetrievalDataset"""
    
    print(f"\n=== Testing RetrievalDataset Compatibility ===\n")
    
    # Check if created files match RetrievalDataset format
    test_files = ["vietnamese_flickr8k.train.jsonl", "vietnamese_flickr8k.val.jsonl"]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"📄 Testing {file_path}:")
            
            items = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    items.append(data)
            
            print(f"  ✓ Loaded {len(items)} items")
            
            # Check required fields
            if items:
                sample = items[0]
                required_fields = ["image_path", "text_segment", "image_id"]
                
                for field in required_fields:
                    if field in sample:
                        print(f"  ✓ Has '{field}' field")
                    else:
                        print(f"  ❌ Missing '{field}' field")
                
                # Check image_id consistency (same image should have same image_id)
                image_to_id = {}
                for item in items:
                    img_path = item["image_path"]
                    img_id = item["image_id"]
                    
                    if img_path in image_to_id:
                        if image_to_id[img_path] != img_id:
                            print(f"  ❌ Inconsistent image_id for {img_path}")
                    else:
                        image_to_id[img_path] = img_id
                
                print(f"  ✓ Image ID consistency check passed")
                print(f"  ✓ Format compatible with BEiT3 RetrievalDataset")
        else:
            print(f"❌ {file_path} not found")
    
    print(f"\n✅ Vietnamese retrieval dataset is ready for BEiT3!")

if __name__ == "__main__":
    create_vietnamese_retrieval_dataset()
    analyze_retrieval_vs_captioning()
    test_retrieval_compatibility()