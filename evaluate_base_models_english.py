"""
Evaluate BASE (pretrained) models with English captions
Compare with finetuned Vietnamese models
"""
import torch
import os
import sys
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

# Import OpenCLIP
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'open_clip', 'src'))
import open_clip

# Import BEiT3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'beit3'))
from timm.models import create_model
import modeling_finetune
import utils as beit3_utils

# ======================
# Configuration
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
test_csv = r"D:\Projects\doan\data\en_captions_test.csv"
test_images_folder = r"D:\Projects\kaggle\test"
beit3_base_model = r"D:\Projects\doan\beit3\beit3_base_itc_patch16_224.pth"
beit3_spm = r"D:\Projects\doan\beit3\beit3.spm"

print(f"Device: {device}")
print(f"Test CSV: {test_csv}")
print(f"Test images: {test_images_folder}")


def load_test_data(csv_path):
    """
    Load test data from CSV
    Returns: dict mapping image_name -> list of (vi_caption, en_caption) tuples
    """
    print(f"\nLoading test data from {csv_path}...")
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    
    # Group by image
    test_data = defaultdict(list)
    for _, row in df.iterrows():
        image_name = os.path.basename(row['image_filename'])
        test_data[image_name].append({
            'vi_caption': row['caption'],
            'en_caption': row['en_caption']
        })
    
    print(f"✓ Loaded {len(test_data)} unique images with {len(df)} total caption pairs")
    return test_data


def load_openclip_base():
    """Load OpenCLIP base model (pretrained on English)"""
    print("\nLoading OpenCLIP base model (pretrained)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name='ViT-B-16',
        pretrained='datacomp_xl_s13b_b90k'  # Use original OpenAI weights
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    
    print("✓ OpenCLIP base model loaded")
    return model, preprocess, tokenizer


def load_beit3_base():
    """Load BEiT3 base model (pretrained for retrieval)"""
    print("\nLoading BEiT3 base model (pretrained)...")
    
    # Load tokenizer
    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer(beit3_spm)
    
    # Create retrieval model with correct architecture
    model = create_model(
        'beit3_base_patch16_224_retrieval',
        pretrained=False,
        drop_path_rate=0.0,
        vocab_size=64010,
        checkpoint_activations=None
    )
    
    # Load pretrained weights
    if os.path.exists(beit3_base_model):
        checkpoint = torch.load(beit3_base_model, map_location='cpu', weights_only=False)
        # Load with may_interpolate to handle position embeddings
        beit3_utils.load_model_and_may_interpolate(
            ckpt_path=beit3_base_model,
            model=model,
            model_key='model',
            model_prefix=''
        )
        print(f"✓ Loaded base weights from {beit3_base_model}")
    else:
        print(f"Warning: Base model not found at {beit3_base_model}")
        print("   BEiT3 evaluation will be skipped")
        return None, None, None
    
    model = model.to(device).eval()
    
    # Create transform
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])
    
    print("✓ BEiT3 base model loaded")
    return model, preprocess, tokenizer


def encode_image_openclip(model, preprocess, image_path):
    """Encode single image with OpenCLIP"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def encode_text_openclip(model, tokenizer, text):
    """Encode text with OpenCLIP"""
    tokens = tokenizer([text]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy()[0]


def encode_image_beit3(model, preprocess, image_path):
    """Encode single image with BEiT3"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            vision_cls, _ = model(image=image_tensor, only_infer=True)
        
        return vision_cls.cpu().numpy()[0]
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def encode_text_beit3(model, tokenizer, text, max_len=64):
    """Encode text with BEiT3"""
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    if len(token_ids) > max_len - 2:
        token_ids = token_ids[:max_len - 2]
    
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    language_tokens = [bos_token_id] + token_ids + [eos_token_id]
    num_tokens = len(language_tokens)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
    language_tokens = language_tokens + [pad_token_id] * (max_len - num_tokens)
    
    language_tokens = torch.tensor([language_tokens]).to(device)
    padding_mask = torch.tensor([padding_mask]).to(device)
    
    with torch.no_grad():
        _, language_cls = model(text_description=language_tokens, padding_mask=padding_mask, only_infer=True)
    
    return language_cls.cpu().numpy()[0]


def calculate_similarity(text_emb, image_emb):
    """Calculate cosine similarity"""
    return np.dot(text_emb, image_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(image_emb))


def reciprocal_rank_fusion(similarities1, similarities2, alpha=0.5, k=60):
    """
    Reciprocal Rank Fusion
    
    Args:
        similarities1: List of (image_name, similarity) from model 1
        similarities2: List of (image_name, similarity) from model 2
        alpha: Weight for model 1 (1-alpha for model 2)
        k: RRF constant (default 60)
    
    Returns:
        List of (image_name, fused_score) sorted by score
    """
    # Create rank-based scores
    fusion_scores = {}
    
    # Add scores from model 1
    for rank, (img_name, _) in enumerate(similarities1):
        rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed
        fusion_scores[img_name] = alpha * rrf_score
    
    # Add scores from model 2
    for rank, (img_name, _) in enumerate(similarities2):
        rrf_score = 1.0 / (k + rank + 1)
        if img_name in fusion_scores:
            fusion_scores[img_name] += (1 - alpha) * rrf_score
        else:
            fusion_scores[img_name] = (1 - alpha) * rrf_score
    
    # Sort by fused score
    fused_results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    return fused_results


def evaluate_model(model_name, text_encoder, image_encoder, test_data, images_folder):
    """
    Evaluate a single model
    
    Args:
        model_name: Name for display
        text_encoder: Function(text) -> embedding
        image_encoder: Function(image_path) -> embedding
        test_data: Dict of image_name -> captions
        images_folder: Path to test images
    
    Returns:
        metrics dict
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    ranks = []
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    
    # Pre-encode all images
    print("Pre-encoding all images...")
    image_embeddings = {}
    image_names = list(test_data.keys())
    
    for image_name in tqdm(image_names, desc="Encoding images"):
        image_path = os.path.join(images_folder, image_name)
        if os.path.exists(image_path):
            emb = image_encoder(image_path)
            if emb is not None:
                image_embeddings[image_name] = emb
    
    print(f"✓ Encoded {len(image_embeddings)} images")
    
    # Evaluate each query
    print("Evaluating queries...")
    total_queries = 0
    
    for gt_image_name, captions in tqdm(test_data.items(), desc="Processing queries"):
        if gt_image_name not in image_embeddings:
            continue
        
        for caption_pair in captions:
            # Use English caption for base models
            query_text = caption_pair['en_caption']
            
            # Encode query
            query_emb = text_encoder(query_text)
            
            # Calculate similarities with all images
            similarities = []
            for img_name, img_emb in image_embeddings.items():
                sim = calculate_similarity(query_emb, img_emb)
                similarities.append((img_name, sim))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Find rank of ground truth image
            rank = None
            for i, (img_name, _) in enumerate(similarities):
                if img_name == gt_image_name:
                    rank = i + 1  # 1-indexed
                    break
            
            if rank is None:
                rank = len(similarities) + 1
            
            ranks.append(rank)
            total_queries += 1
            
            # Update recalls
            if rank <= 1:
                recall_at_1 += 1
            if rank <= 5:
                recall_at_5 += 1
            if rank <= 10:
                recall_at_10 += 1
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'R@1': (recall_at_1 / total_queries) * 100,
        'R@5': (recall_at_5 / total_queries) * 100,
        'R@10': (recall_at_10 / total_queries) * 100,
        'Mean Rank': np.mean(ranks),
        'Median Rank': np.median(ranks),
        'Total Queries': total_queries
    }
    
    return metrics


def evaluate_fusion(
    model_name,
    text_encoder1,
    image_encoder1,
    text_encoder2,
    image_encoder2,
    test_data,
    images_folder,
    alpha=0.5
):
    """
    Evaluate fusion of two models using Reciprocal Rank Fusion
    
    Args:
        model_name: Name for display
        text_encoder1: Function(text) -> embedding for model 1
        image_encoder1: Function(image_path) -> embedding for model 1
        text_encoder2: Function(text) -> embedding for model 2
        image_encoder2: Function(image_path) -> embedding for model 2
        test_data: Dict of image_name -> captions
        images_folder: Path to test images
        alpha: Weight for model 1 (1-alpha for model 2)
    
    Returns:
        metrics dict
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Fusion weights: {alpha*100:.1f}% Model1 + {(1-alpha)*100:.1f}% Model2")
    print(f"{'='*60}")
    
    ranks = []
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    
    # Pre-encode all images with both models
    print("Pre-encoding all images with both models...")
    image_embeddings1 = {}
    image_embeddings2 = {}
    image_names = list(test_data.keys())
    
    for image_name in tqdm(image_names, desc="Encoding images (Model 1)"):
        image_path = os.path.join(images_folder, image_name)
        if os.path.exists(image_path):
            emb = image_encoder1(image_path)
            if emb is not None:
                image_embeddings1[image_name] = emb
    
    for image_name in tqdm(image_names, desc="Encoding images (Model 2)"):
        image_path = os.path.join(images_folder, image_name)
        if os.path.exists(image_path):
            emb = image_encoder2(image_path)
            if emb is not None:
                image_embeddings2[image_name] = emb
    
    print(f"✓ Encoded {len(image_embeddings1)} images (Model 1), {len(image_embeddings2)} images (Model 2)")
    
    # Evaluate each query
    print("Evaluating queries with fusion...")
    total_queries = 0
    
    for gt_image_name, captions in tqdm(test_data.items(), desc="Processing queries"):
        if gt_image_name not in image_embeddings1 or gt_image_name not in image_embeddings2:
            continue
        
        for caption_pair in captions:
            # Use English caption for base models
            query_text = caption_pair['en_caption']
            
            # Encode query with both models
            query_emb1 = text_encoder1(query_text)
            query_emb2 = text_encoder2(query_text)
            
            # Calculate similarities with all images (Model 1)
            similarities1 = []
            for img_name, img_emb in image_embeddings1.items():
                sim = calculate_similarity(query_emb1, img_emb)
                similarities1.append((img_name, sim))
            similarities1.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate similarities with all images (Model 2)
            similarities2 = []
            for img_name, img_emb in image_embeddings2.items():
                sim = calculate_similarity(query_emb2, img_emb)
                similarities2.append((img_name, sim))
            similarities2.sort(key=lambda x: x[1], reverse=True)
            
            # Apply Reciprocal Rank Fusion
            fused_results = reciprocal_rank_fusion(similarities1, similarities2, alpha=alpha)
            
            # Find rank of ground truth image
            rank = None
            for i, (img_name, _) in enumerate(fused_results):
                if img_name == gt_image_name:
                    rank = i + 1  # 1-indexed
                    break
            
            if rank is None:
                rank = len(fused_results) + 1
            
            ranks.append(rank)
            total_queries += 1
            
            # Update recalls
            if rank <= 1:
                recall_at_1 += 1
            if rank <= 5:
                recall_at_5 += 1
            if rank <= 10:
                recall_at_10 += 1
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'R@1': (recall_at_1 / total_queries) * 100,
        'R@5': (recall_at_5 / total_queries) * 100,
        'R@10': (recall_at_10 / total_queries) * 100,
        'Mean Rank': np.mean(ranks),
        'Median Rank': np.median(ranks),
        'Total Queries': total_queries
    }
    
    return metrics


def main():
    print("="*60)
    print("BASELINE EVALUATION - BASE MODELS (English)")
    print("="*60)
    
    # Load test data
    test_data = load_test_data(test_csv)
    
    # Load models
    openclip_model, openclip_preprocess, openclip_tokenizer = load_openclip_base()
    beit3_model, beit3_preprocess, beit3_tokenizer = load_beit3_base()
    
    # Prepare encoder functions
    openclip_text_encoder = lambda text: encode_text_openclip(openclip_model, openclip_tokenizer, text)
    openclip_image_encoder = lambda img_path: encode_image_openclip(openclip_model, openclip_preprocess, img_path)
    
    # Evaluate both models
    results = []
    
    # 1. OpenCLIP base
    metrics_openclip = evaluate_model(
        "OpenCLIP Base (English)",
        openclip_text_encoder,
        openclip_image_encoder,
        test_data,
        test_images_folder
    )
    results.append(metrics_openclip)
    
    # 2. BEiT3 base (if loaded successfully)
    if beit3_model is not None:
        beit3_text_encoder = lambda text: encode_text_beit3(beit3_model, beit3_tokenizer, text)
        beit3_image_encoder = lambda img_path: encode_image_beit3(beit3_model, beit3_preprocess, img_path)
        
        metrics_beit3 = evaluate_model(
            "BEiT3 Base (English)",
            beit3_text_encoder,
            beit3_image_encoder,
            test_data,
            test_images_folder
        )
        results.append(metrics_beit3)
        
        # 3. Fusion: 39% OpenCLIP + 61% BEiT3
        metrics_fusion = evaluate_fusion(
            "Fusion (39% OpenCLIP + 61% BEiT3)",
            openclip_text_encoder,
            openclip_image_encoder,
            beit3_text_encoder,
            beit3_image_encoder,
            test_data,
            test_images_folder,
            alpha=0.45
        )
        results.append(metrics_fusion)
    else:
        print("\n⚠️  BEiT3 base evaluation skipped (model not available)")
        print("⚠️  Fusion evaluation skipped (requires both models)")
    
    # Print results table
    print("\n" + "="*80)
    print("BASELINE EVALUATION RESULTS")
    print("="*80)
    print(f"{'Model':<30} {'R@1':>10} {'R@5':>10} {'R@10':>10} {'Mean Rank':>12} {'Median Rank':>12}")
    print("-"*80)
    
    for metrics in results:
        print(f"{metrics['Model']:<30} {metrics['R@1']:>9.2f}% {metrics['R@5']:>9.2f}% "
              f"{metrics['R@10']:>9.2f}% {metrics['Mean Rank']:>11.2f} {metrics['Median Rank']:>12.1f}")
    
    print("="*80)
    print(f"Total queries evaluated: {results[0]['Total Queries']}")
    print("="*80)
    
    # Save results
    import json
    output_file = "baseline_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
