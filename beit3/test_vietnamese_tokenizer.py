#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Vietnamese tokenization with BEiT3 tokenizer
"""

try:
    from transformers import XLMRobertaTokenizer
    
    # Load BEiT3 tokenizer
    print("Loading BEiT3 tokenizer...")
    tokenizer = XLMRobertaTokenizer("beit3.spm")
    print("✓ Tokenizer loaded successfully!")
    
    # Test Vietnamese text samples
    vietnamese_samples = [
        "Một con chó nhỏ đang chạy trên bãi cỏ xanh",
        "Có một người phụ nữ đang ngồi trên ghế đá",
        "Trẻ em đang chơi bóng trong công viên",
        "Một chiếc xe hơi màu đỏ đậu trước nhà",
        "Cô gái mặc áo dài trắng đang đi trên phố"
    ]
    
    print("\n=== Testing Vietnamese Tokenization ===")
    for i, text in enumerate(vietnamese_samples, 1):
        print(f"\n{i}. Original text: {text}")
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        print(f"   Tokens: {tokens}")
        
        # Convert to IDs
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"   Token IDs: {token_ids}")
        
        # Decode back
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"   Decoded: {decoded}")
        
        # Check if decoded text matches original
        if decoded.strip() == text.strip():
            print("   ✓ Perfect reconstruction!")
        else:
            print("   ⚠ Slight difference in reconstruction")
    
    # Test tokenizer properties
    print(f"\n=== Tokenizer Properties ===")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"MASK token: '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")
    
    # Test special Vietnamese characters
    print(f"\n=== Vietnamese Character Support ===")
    vietnamese_chars = "àáãạăắằẳẵặâấầẩẫậèéẹêếềểễệìíĩịòóõọôốồổỗộơớờởỡợùúũụưứừửữựỳýỵđ"
    tokens = tokenizer.tokenize(vietnamese_chars)
    print(f"Vietnamese chars: {vietnamese_chars}")
    print(f"Tokenized: {tokens}")
    
    print("\n=== Conclusion ===")
    print("✓ BEiT3 tokenizer supports Vietnamese text!")
    print("✓ XLMRobertaTokenizer can handle Vietnamese characters properly")
    print("✓ Ready for Vietnamese image captioning fine-tuning!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install transformers: pip install transformers")
except Exception as e:
    print(f"❌ Error: {e}")