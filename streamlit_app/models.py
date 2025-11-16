"""
Model management and loading utilities
"""
import streamlit as st
import torch
import open_clip
from PIL import Image
import sys
import os

# Add parent directory to path for BEiT3 imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'beit3'))

from config import (
    OPENCLIP_MODEL_PATH,
    BEIT3_TOKENIZER_PATH,
    BEIT3_CHECKPOINT_PATH,
    DEVICE
)


class ModelManager:
    """Manages loading and caching of OpenCLIP and BEiT3 models"""
    
    def __init__(self):
        self.device = DEVICE
    
    @st.cache_resource
    def load_openclip(_self):
        """Load OpenCLIP model with checkpoint (same as rank_fusion_encode.py)"""
        try:
            # Load model with checkpoint directly - OpenCLIP handles checkpoint format automatically
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name='ViT-B-16',
                pretrained=OPENCLIP_MODEL_PATH
            )
            
            model = model.to(_self.device)
            model.eval()
            
            # Get tokenizer
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            
            print(f"✓ Loaded OpenCLIP from {OPENCLIP_MODEL_PATH}")
            return model, preprocess, tokenizer
        except Exception as e:
            st.error(f"Error loading OpenCLIP model: {str(e)}")
            raise
    
    @st.cache_resource
    def load_beit3(_self):
        """Load BEiT3 model with checkpoint (same as rank_fusion_encode.py)"""
        try:
            from transformers import XLMRobertaTokenizer
            from torchvision import transforms
            from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
            from timm.models import create_model
            import utils as beit3_utils
            
            # Load tokenizer (same as rank_fusion_demo.py)
            tokenizer = XLMRobertaTokenizer(BEIT3_TOKENIZER_PATH)
            
            # Create model (same way as rank_fusion_encode.py)
            model = create_model(
                'beit3_base_patch16_224_retrieval',
                pretrained=False,
                drop_path_rate=0.1,
                vocab_size=64010,
                checkpoint_activations=None
            )
            
            # Load checkpoint using beit3_utils (handles interpolation automatically)
            if os.path.exists(BEIT3_CHECKPOINT_PATH):
                beit3_utils.load_model_and_may_interpolate(
                    ckpt_path=BEIT3_CHECKPOINT_PATH,
                    model=model,
                    model_key='model|module',
                    model_prefix=''
                )
                print(f"✓ Loaded BEiT3 from {BEIT3_CHECKPOINT_PATH}")
            else:
                print(f"Warning: Checkpoint not found at {BEIT3_CHECKPOINT_PATH}")
            
            model = model.to(_self.device)
            model.eval()
            
            # Create image transform (same as rank_fusion_encode.py)
            image_transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=3), 
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
            ])
            
            print(f"✓ BEiT3 loaded on {_self.device}")
            
            # Return model, tokenizer, and transform
            return model, tokenizer, image_transform
        except Exception as e:
            st.error(f"Error loading BEiT3 model: {str(e)}")
            raise
    
    def get_openclip(self):
        """Get cached OpenCLIP model or load if not cached"""
        if 'openclip_model' not in st.session_state:
            with st.spinner("Loading OpenCLIP model..."):
                model, preprocess, tokenizer = self.load_openclip()
                st.session_state.openclip_model = model
                st.session_state.openclip_preprocess = preprocess
                st.session_state.openclip_tokenizer = tokenizer
        
        return (
            st.session_state.openclip_model,
            st.session_state.openclip_preprocess,
            st.session_state.openclip_tokenizer
        )
    
    def get_beit3(self):
        """Get cached BEiT3 model or load if not cached"""
        if 'beit3_model' not in st.session_state:
            with st.spinner("Loading BEiT3 model..."):
                model, tokenizer, image_transform = self.load_beit3()
                st.session_state.beit3_model = model
                st.session_state.beit3_tokenizer = tokenizer
                st.session_state.beit3_transform = image_transform
        
        return (
            st.session_state.beit3_model,
            st.session_state.beit3_tokenizer,
            st.session_state.beit3_transform
        )
