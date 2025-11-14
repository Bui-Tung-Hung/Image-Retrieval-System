"""
Test script to check SigLIP2 model availability and specs
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'open_clip', 'src'))

import torch
import open_clip

print("="*60)
print("TESTING SigLIP2 MODEL")
print("="*60)

# Check available SigLIP2 models
print("\nAvailable SigLIP2 models:")
models = open_clip.list_pretrained()
siglip2_models = [m for m in models if 'siglip2' in m[0].lower() and 'so400m' in m[0].lower()]
for idx, (model_name, pretrained) in enumerate(siglip2_models, 1):
    print(f"  {idx}. {model_name:40} ({pretrained})")

# Test loading ViT-SO400M-16-SigLIP2-384
print("\n" + "="*60)
print("Loading ViT-SO400M-16-SigLIP2-384...")
print("="*60)

try:
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-SO400M-16-SigLIP2-384',
        pretrained='webli'
    )
    
    print("✓ Model loaded successfully!")
    print(f"\nModel Architecture:")
    print(f"  - Vision tower: {model.visual.__class__.__name__}")
    print(f"  - Text tower: {model.text.__class__.__name__}")
    
    # Get embedding dimensions
    if hasattr(model.visual, 'output_dim'):
        vision_dim = model.visual.output_dim
    elif hasattr(model, 'embed_dim'):
        vision_dim = model.embed_dim
    else:
        # Try to infer from a dummy forward pass
        dummy_img = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            vision_features = model.encode_image(dummy_img)
        vision_dim = vision_features.shape[-1]
    
    if hasattr(model.text, 'output_dim'):
        text_dim = model.text.output_dim
    elif hasattr(model, 'embed_dim'):
        text_dim = model.embed_dim
    else:
        # Try to infer from a dummy forward pass
        tokenizer = open_clip.get_tokenizer('ViT-SO400M-16-SigLIP2-384')
        dummy_text = tokenizer(["test"])
        with torch.no_grad():
            text_features = model.encode_text(dummy_text)
        text_dim = text_features.shape[-1]
    
    print(f"\nEmbedding Dimensions:")
    print(f"  - Vision embedding: {vision_dim} dims")
    print(f"  - Text embedding: {text_dim} dims")
    
    # Test encode
    print(f"\nTesting encoding...")
    from PIL import Image
    import numpy as np
    
    # Create a dummy image
    dummy_pil = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
    img_tensor = preprocess(dummy_pil).unsqueeze(0)
    
    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)
        txt_emb = model.encode_text(tokenizer(["test image"]))
    
    print(f"  ✓ Image encoding: {img_emb.shape}")
    print(f"  ✓ Text encoding: {txt_emb.shape}")
    
    print("\n" + "="*60)
    print("✅ SigLIP2 MODEL IS READY FOR EVALUATION!")
    print("="*60)
    
    print(f"\nComparison with existing models:")
    print(f"  - OpenCLIP ViT-B-16:  512 dims")
    print(f"  - BEiT3:              768 dims")
    print(f"  - SigLIP2 SO400M:     {vision_dim} dims")
    
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
