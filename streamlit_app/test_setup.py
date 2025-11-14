"""
Test script to verify models and FAISS setup
Run this before launching the Streamlit app
"""
import sys
import os

# Add beit3 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'beit3'))

print("=" * 60)
print("TESTING IMAGE RETRIEVAL SYSTEM")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/6] Testing module imports...")
try:
    import config
    import models
    import faiss_manager
    import image_encoder
    import search_engine
    import ui_components
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Check config paths
print("\n[2/6] Checking configuration paths...")
from config import (
    OPENCLIP_MODEL_PATH,
    BEIT3_MODEL_PATH,
    BEIT3_TOKENIZER_PATH,
    BEIT3_CHECKPOINT_PATH
)

paths_ok = True
if not os.path.exists(OPENCLIP_MODEL_PATH):
    print(f"⚠️  OpenCLIP checkpoint not found: {OPENCLIP_MODEL_PATH}")
    paths_ok = False
else:
    print(f"✅ OpenCLIP checkpoint found")

if not os.path.exists(BEIT3_MODEL_PATH):
    print(f"⚠️  BEiT3 base weights not found: {BEIT3_MODEL_PATH}")
    paths_ok = False
else:
    print(f"✅ BEiT3 base weights found")

if not os.path.exists(BEIT3_TOKENIZER_PATH):
    print(f"❌ BEiT3 tokenizer not found: {BEIT3_TOKENIZER_PATH}")
    paths_ok = False
    sys.exit(1)
else:
    print(f"✅ BEiT3 tokenizer found")

if not os.path.exists(BEIT3_CHECKPOINT_PATH):
    print(f"⚠️  BEiT3 finetuned checkpoint not found: {BEIT3_CHECKPOINT_PATH}")
    print("   Will use base weights only")
else:
    print(f"✅ BEiT3 finetuned checkpoint found")

# Test 3: Check GPU availability
print("\n[3/6] Checking GPU availability...")
import torch
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  CUDA not available, will use CPU (slower)")

# Test 4: Check FAISS
print("\n[4/6] Checking FAISS...")
try:
    import faiss
    print(f"✅ FAISS version: {faiss.__version__}")
    
    # Test IndexIDMap2
    test_index = faiss.IndexIDMap2(faiss.IndexFlatIP(128))
    print("✅ IndexIDMap2 supported")
except Exception as e:
    print(f"❌ FAISS error: {e}")
    sys.exit(1)

# Test 5: Test BEiT3 imports
print("\n[5/6] Testing BEiT3 imports...")
try:
    from modeling_finetune import beit3_base_patch16_224_retrieval
    from transformers import XLMRobertaTokenizer
    from torchvision import transforms
    from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    print("✅ BEiT3 modules imported successfully")
except Exception as e:
    print(f"❌ BEiT3 import error: {e}")
    print("   Make sure you're in the streamlit_app directory")
    sys.exit(1)

# Test 6: Test OpenCLIP
print("\n[6/6] Testing OpenCLIP imports...")
try:
    import open_clip
    print("✅ OpenCLIP imported successfully")
except Exception as e:
    print(f"❌ OpenCLIP import error: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("SETUP VERIFICATION COMPLETE")
print("=" * 60)
print("\nYou can now run the Streamlit app:")
print("  streamlit run app.py")
print("\n" + "=" * 60)
