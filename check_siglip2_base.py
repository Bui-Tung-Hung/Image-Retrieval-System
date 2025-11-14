import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'open_clip', 'src'))
import open_clip

print("Checking SigLIP2 Base models in OpenCLIP...")
models = open_clip.list_pretrained()
siglip2_models = [m for m in models if 'siglip2' in m[0].lower()]

print("\nAll SigLIP2 models:")
for name, pretrained in siglip2_models:
    print(f"  {name:50} {pretrained}")

print("\nBase models specifically:")
base_models = [m for m in siglip2_models if 'base' in m[0].lower()]
for name, pretrained in base_models:
    print(f"  {name:50} {pretrained}")
