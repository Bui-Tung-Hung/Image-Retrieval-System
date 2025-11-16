"""
Configuration file for Streamlit Image Retrieval App
"""
import os
import torch

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDICES_DIR = os.path.join(DATA_DIR, "indices")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")

# Model paths
OPENCLIP_MODEL_PATH = "d:/Projects/doan/open_clip/checkpoints/epoch_15.pt"
BEIT3_MODEL_PATH = "d:/Projects/doan/beit3/beit3_base_itc_patch16_224.pth"
BEIT3_TOKENIZER_PATH = "d:/Projects/doan/beit3/beit3.spm"
BEIT3_CHECKPOINT_PATH = "d:/Projects/doan/beit3/ckpt/checkpoint-best.pth"

# Model dimensions
OPENCLIP_DIM = 512
BEIT3_DIM = 768

# FAISS index files
OPENCLIP_INDEX_FILE = os.path.join(INDICES_DIR, "openclip.index")
BEIT3_INDEX_FILE = os.path.join(INDICES_DIR, "beit3.index")
METADATA_FILE = os.path.join(INDICES_DIR, "metadata.json")

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Encoding settings
BATCH_SIZE = 32
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

# Search settings
DEFAULT_TOP_K = 20
DEFAULT_FUSION_WEIGHT = 0.5
# FUSION_K = 60  # Removed - now using score-based fusion instead of RRF

# UI settings
GRID_COLUMNS = 4
MAX_FILE_SIZE_MB = 10
