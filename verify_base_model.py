# import sys
# import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'open_clip', 'src'))
# import open_clip
# import torch

# print("Loading SigLIP2 Base model...")
# model, _, preprocess = open_clip.create_model_and_transforms(
#     'ViT-B-16-SigLIP2-256',
#     pretrained='webli',
#     device='cpu'
# )

# print("\nModel architecture:")
# print(f"  Model name: ViT-B-16-SigLIP2-256")
# print(f"  Pretrained: webli")

# # Check embedding dimension
# dummy_input = torch.randn(1, 3, 256, 256)
# with torch.no_grad():
#     output = model.encode_image(dummy_input)
    
# print(f"\nEmbedding dimension: {output.shape[1]}")
# print(f"Image size: 256x256")

# # Count parameters
# total_params = sum(p.numel() for p in model.parameters())
# print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.1f}M)")

# # Test tokenizer
# try:
#     tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP2-256')
#     print(f"\n✅ Tokenizer loaded successfully")
#     print(f"   Tokenizer type: {type(tokenizer)}")
# except Exception as e:
#     print(f"\n❌ Tokenizer error: {e}")
import open_clip
for i in open_clip.list_pretrained():
    print(i)