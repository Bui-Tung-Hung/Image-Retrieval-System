import torch
from PIL import Image
import os
import sys
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model

# Import OpenCLIP
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'open_clip', 'src'))
import open_clip

# Import BEiT3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'beit3'))
import modeling_finetune
import utils as beit3_utils

# ======================
# Configuration
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32  # Reduced for stability with 2 models
test_folder = r"D:\Projects\kaggle\test"
openclip_ckpt = r"D:\Projects\doan\open_clip\checkpoints\epoch_15.pt"
beit3_ckpt = r"C:\Users\LAPTOP\Downloads\BEiT3\ckpt\checkpoint-best.pth"
beit3_spm = r"D:\Projects\doan\beit3\beit3.spm"
output_folder = r"D:\Projects\doan\rank_fusion_output"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

print(f"Device: {device}")
print(f"Output folder: {output_folder}")


def load_openclip_model():
    """Load OpenCLIP ViT-B-16 model"""
    print("\n[1/2] Loading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name='ViT-B-16',
        pretrained=openclip_ckpt
    )
    model = model.to(device)
    model.eval()
    print(f"✓ OpenCLIP loaded on {device}")
    return model, preprocess


def load_beit3_model():
    """Load BEiT3 retrieval model"""
    print("\n[2/2] Loading BEiT3 model...")
    model = create_model(
        'beit3_base_patch16_224_retrieval',
        pretrained=False,
        drop_path_rate=0.1,
        vocab_size=64010,
        checkpoint_activations=None
    )
    
    if os.path.exists(beit3_ckpt):
        beit3_utils.load_model_and_may_interpolate(
            ckpt_path=beit3_ckpt,
            model=model,
            model_key='model|module',
            model_prefix=''
        )
        print(f"✓ Loaded checkpoint from {beit3_ckpt}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {beit3_ckpt}")
    
    model = model.to(device)
    model.eval()
    
    # Create BEiT3 transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])
    
    print(f"✓ BEiT3 loaded on {device}")
    return model, preprocess


def encode_images_openclip(model, preprocess, image_list):
    """Encode all images using OpenCLIP"""
    print(f"\n[OpenCLIP] Encoding {len(image_list)} images...")
    all_features = []
    
    for i in tqdm(range(0, len(image_list), batch_size), desc="OpenCLIP encoding"):
        batch_paths = image_list[i:i+batch_size]
        
        try:
            batch_images = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                batch_images.append(preprocess(img))
            
            batch_tensor = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                batch_features = model.encode_image(batch_tensor)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)
            
            all_features.append(batch_features.cpu())
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            continue
    
    all_features = torch.cat(all_features, dim=0)
    print(f"✓ OpenCLIP embeddings shape: {all_features.shape}")
    return all_features


def encode_images_beit3(model, preprocess, image_list):
    """Encode all images using BEiT3"""
    print(f"\n[BEiT3] Encoding {len(image_list)} images...")
    all_features = []
    
    for i in tqdm(range(0, len(image_list), batch_size), desc="BEiT3 encoding"):
        batch_paths = image_list[i:i+batch_size]
        
        try:
            batch_images = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                batch_images.append(preprocess(img))
            
            batch_tensor = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                vision_cls, _ = model(image=batch_tensor, only_infer=True)
                batch_features = vision_cls
            
            all_features.append(batch_features.cpu())
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            continue
    
    all_features = torch.cat(all_features, dim=0)
    print(f"✓ BEiT3 embeddings shape: {all_features.shape}")
    return all_features


def main():
    print("="*60)
    print("RANK FUSION - IMAGE ENCODING PIPELINE")
    print("="*60)
    
    # Load models
    openclip_model, openclip_preprocess = load_openclip_model()
    beit3_model, beit3_preprocess = load_beit3_model()
    
    # Scan images
    print(f"\nScanning images from {test_folder}...")
    image_list = [
        os.path.join(test_folder, img)
        for img in sorted(os.listdir(test_folder))
        if img.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    print(f"✓ Found {len(image_list)} images")
    
    # Encode with OpenCLIP
    openclip_embeddings = encode_images_openclip(openclip_model, openclip_preprocess, image_list)
    
    # Encode with BEiT3
    beit3_embeddings = encode_images_beit3(beit3_model, beit3_preprocess, image_list)
    
    # Save embeddings
    print("\nSaving embeddings...")
    torch.save(openclip_embeddings, os.path.join(output_folder, "openclip_embeddings.pt"))
    torch.save(beit3_embeddings, os.path.join(output_folder, "beit3_embeddings.pt"))
    print("✓ Saved raw embeddings")
    
    # Create FAISS indices
    print("\nCreating FAISS indices...")
    
    # OpenCLIP index
    openclip_np = openclip_embeddings.numpy().astype('float32')
    faiss.normalize_L2(openclip_np)
    index_openclip = faiss.IndexFlatIP(openclip_np.shape[1])
    index_openclip.add(openclip_np)
    faiss.write_index(index_openclip, os.path.join(output_folder, "openclip_image_index.faiss"))
    print(f"✓ OpenCLIP FAISS index: {index_openclip.ntotal} vectors, dim={openclip_np.shape[1]}")
    
    # BEiT3 index
    beit3_np = beit3_embeddings.numpy().astype('float32')
    faiss.normalize_L2(beit3_np)
    index_beit3 = faiss.IndexFlatIP(beit3_np.shape[1])
    index_beit3.add(beit3_np)
    faiss.write_index(index_beit3, os.path.join(output_folder, "beit3_image_index.faiss"))
    print(f"✓ BEiT3 FAISS index: {index_beit3.ntotal} vectors, dim={beit3_np.shape[1]}")
    
    # Save image paths
    with open(os.path.join(output_folder, "image_paths.pkl"), "wb") as f:
        pickle.dump(image_list, f)
    print(f"✓ Saved {len(image_list)} image paths")
    
    print("\n" + "="*60)
    print("✅ ENCODING COMPLETE!")
    print("="*60)
    print(f"Output files in: {output_folder}")
    print(f"  - openclip_embeddings.pt")
    print(f"  - beit3_embeddings.pt")
    print(f"  - openclip_image_index.faiss")
    print(f"  - beit3_image_index.faiss")
    print(f"  - image_paths.pkl")
    print("="*60)


if __name__ == "__main__":
    main()
