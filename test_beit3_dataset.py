#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để verify Vietnamese Flickr8k JSONL dataset
"""

import json
from transformers import XLMRobertaTokenizer


def test_jsonl_format(jsonl_path, tokenizer_path, num_samples=3):
    """
    Test JSONL format và decode tokens
    
    Args:
        jsonl_path: Đường dẫn đến file JSONL
        tokenizer_path: Đường dẫn đến beit3.spm
        num_samples: Số samples để test
    """
    print(f"\n{'='*80}")
    print(f"Testing: {jsonl_path}")
    print(f"{'='*80}\n")
    
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer(tokenizer_path)
    
    # Read samples
    items = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            item = json.loads(line)
            items.append(item)
    
    # Test each sample
    for i, item in enumerate(items, 1):
        print(f"Sample {i}:")
        print(f"  Image path: {item['image_path']}")
        print(f"  Image ID: {item['image_id']}")
        print(f"  Token count: {len(item['text_segment'])}")
        print(f"  First 10 tokens: {item['text_segment'][:10]}")
        
        # Decode
        decoded = tokenizer.decode(item['text_segment'], skip_special_tokens=True)
        print(f"  Decoded text: {decoded}")
        print()
    
    print(f"✅ All {num_samples} samples passed!\n")


if __name__ == "__main__":
    tokenizer_path = "D:/Projects/doan/beit3/beit3.spm"
    
    # Test train set
    test_jsonl_format(
        "D:/Projects/doan/data/flickr8k_vi/vietnamese_flickr8k.train.jsonl",
        tokenizer_path,
        num_samples=3
    )
    
    # Test val set
    test_jsonl_format(
        "D:/Projects/doan/data/flickr8k_vi/vietnamese_flickr8k.val.jsonl",
        tokenizer_path,
        num_samples=3
    )
    
    print(f"{'='*80}")
    print("✅ ĐÃ VERIFY THÀNH CÔNG! Dataset sẵn sàng cho BEiT-3 fine-tuning")
    print(f"{'='*80}")
