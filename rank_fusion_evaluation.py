import torch
import os
import sys
import pickle
import numpy as np
import faiss
import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
from transformers import XLMRobertaTokenizer

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
output_folder = r"D:\Projects\doan\rank_fusion_output"
ground_truth_csv = r"D:\Projects\kaggle\test_corrected.csv"
openclip_ckpt = r"D:\Projects\doan\open_clip\checkpoints\epoch_15.pt"
beit3_ckpt = r"C:\Users\LAPTOP\Downloads\BEiT3\ckpt\checkpoint-best.pth"
beit3_spm = r"D:\Projects\doan\beit3\beit3.spm"


def load_ground_truth(csv_path):
    """Load ground truth from CSV: image_filename;caption"""
    print(f"\nLoading ground truth from {csv_path}...")
    df = pd.read_csv(csv_path, sep=';', names=['image_filename', 'caption'], encoding='utf-8')
    
    # Group captions by image
    gt_dict = defaultdict(list)
    for _, row in df.iterrows():
        image_name = os.path.basename(row['image_filename'])
        gt_dict[image_name].append(row['caption'])
    
    print(f"✓ Loaded {len(gt_dict)} unique images with {len(df)} total captions")
    return gt_dict


def load_models_and_indices():
    """Load both models, FAISS indices, and image paths"""
    print("\n" + "="*60)
    print("Loading models and indices...")
    print("="*60)
    
    # Load image paths
    with open(os.path.join(output_folder, "image_paths.pkl"), "rb") as f:
        image_paths = pickle.load(f)
    print(f"✓ Loaded {len(image_paths)} image paths")
    
    # Load FAISS indices
    index_openclip = faiss.read_index(os.path.join(output_folder, "openclip_image_index.faiss"))
    index_beit3 = faiss.read_index(os.path.join(output_folder, "beit3_image_index.faiss"))
    print(f"✓ Loaded FAISS indices: OpenCLIP ({index_openclip.ntotal}), BEiT3 ({index_beit3.ntotal})")
    
    # Load OpenCLIP model
    print("\nLoading OpenCLIP model...")
    openclip_model, _, _ = open_clip.create_model_and_transforms(
        model_name='ViT-B-16',
        pretrained=openclip_ckpt
    )
    openclip_model = openclip_model.to(device).eval()
    openclip_tokenizer = open_clip.get_tokenizer('ViT-B-16')
    print("✓ OpenCLIP ready")
    
    # Load BEiT3 model
    print("\nLoading BEiT3 model...")
    beit3_model = create_model(
        'beit3_base_patch16_224_retrieval',
        pretrained=False,
        drop_path_rate=0.1,
        vocab_size=64010,
        checkpoint_activations=None,
    )
    beit3_utils.load_model_and_may_interpolate(
        ckpt_path=beit3_ckpt,
        model=beit3_model,
        model_key='model|module',
        model_prefix=''
    )
    beit3_model = beit3_model.to(device).eval()
    beit3_tokenizer = XLMRobertaTokenizer(beit3_spm)
    print("✓ BEiT3 ready")
    
    return {
        'openclip_model': openclip_model,
        'openclip_tokenizer': openclip_tokenizer,
        'beit3_model': beit3_model,
        'beit3_tokenizer': beit3_tokenizer,
        'index_openclip': index_openclip,
        'index_beit3': index_beit3,
        'image_paths': image_paths
    }


def encode_text_openclip(model, tokenizer, text):
    """Encode text query using OpenCLIP"""
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype('float32')


def encode_text_beit3(model, tokenizer, text, max_len=64):
    """Encode text query using BEiT3"""
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
    
    return language_cls.cpu().numpy().astype('float32')


def search_with_fusion(query_text, models_dict, weight1=0.3, weight2=0.7, top_k=10):
    """
    Search with rank fusion
    weight1: weight for OpenCLIP
    weight2: weight for BEiT3
    """
    # Encode query with both models
    query_emb1 = encode_text_openclip(
        models_dict['openclip_model'],
        models_dict['openclip_tokenizer'],
        query_text
    )
    
    query_emb2 = encode_text_beit3(
        models_dict['beit3_model'],
        models_dict['beit3_tokenizer'],
        query_text
    )
    
    # Normalize
    faiss.normalize_L2(query_emb1)
    faiss.normalize_L2(query_emb2)
    
    # Search both indices
    k = min(top_k * 10, models_dict['index_openclip'].ntotal)  # Get more for fusion
    
    sims1, indices1 = models_dict['index_openclip'].search(query_emb1, k)
    sims2, indices2 = models_dict['index_beit3'].search(query_emb2, k)
    
    # Fusion: collect all unique indices and their scores
    fusion_scores = {}
    for i, idx in enumerate(indices1[0]):
        fusion_scores[idx] = weight1 * sims1[0][i]
    
    for i, idx in enumerate(indices2[0]):
        if idx in fusion_scores:
            fusion_scores[idx] += weight2 * sims2[0][i]
        else:
            fusion_scores[idx] = weight2 * sims2[0][i]
    
    # Sort by fusion score
    sorted_results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Convert to readable format
    results = []
    for idx, score in sorted_results:
        results.append({
            'image_path': models_dict['image_paths'][idx],
            'image_name': os.path.basename(models_dict['image_paths'][idx]),
            'score': float(score),
            'rank': len(results) + 1
        })
    
    return results


def calculate_metrics(ground_truth, all_results, image_paths):
    """
    Calculate R@1, R@5, R@10, Mean Rank, Median Rank
    all_results: list of (query, search_results) tuples
    """
    ranks = []
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    total_queries = 0
    
    for query_text, search_results in all_results:
        # Find ground truth image for this query
        gt_image = None
        for img_name, captions in ground_truth.items():
            if query_text in captions:
                gt_image = img_name
                break
        
        if gt_image is None:
            continue
        
        total_queries += 1
        
        # Find rank of ground truth image
        rank = None
        for i, result in enumerate(search_results):
            if result['image_name'] == gt_image:
                rank = i + 1  # 1-indexed
                break
        
        if rank is None:
            rank = len(image_paths) + 1  # Not found
        
        ranks.append(rank)
        
        # Update recalls
        if rank <= 1:
            recall_at_1 += 1
        if rank <= 5:
            recall_at_5 += 1
        if rank <= 10:
            recall_at_10 += 1
    
    if total_queries == 0:
        return None
    
    metrics = {
        'R@1': (recall_at_1 / total_queries) * 100,
        'R@5': (recall_at_5 / total_queries) * 100,
        'R@10': (recall_at_10 / total_queries) * 100,
        'Mean Rank': np.mean(ranks),
        'Median Rank': np.median(ranks),
        'Total Queries': total_queries
    }
    
    return metrics


def evaluate_model(models_dict, ground_truth, weight1=1.0, weight2=0.0, model_name="Model"):
    """Evaluate a single model or fusion"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Weights: OpenCLIP={weight1:.1f}, BEiT3={weight2:.1f}")
    print(f"{'='*60}")
    
    all_results = []
    
    # Collect all unique queries
    all_queries = []
    for img_name, captions in ground_truth.items():
        all_queries.extend(captions)
    
    print(f"Processing {len(all_queries)} queries...")
    
    for query in tqdm(all_queries, desc=f"Evaluating {model_name}"):
        results = search_with_fusion(query, models_dict, weight1=weight1, weight2=weight2, top_k=10)
        all_results.append((query, results))
    
    # Calculate metrics
    metrics = calculate_metrics(ground_truth, all_results, models_dict['image_paths'])
    
    return metrics


def main():
    print("="*60)
    print("RANK FUSION - EVALUATION")
    print("="*60)
    
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_csv)
    
    # Load models and indices
    models_dict = load_models_and_indices()
    
    # Evaluate all configurations
    results = {}
    
    # # 1. OpenCLIP only
    # results['OpenCLIP Only'] = evaluate_model(
    #     models_dict, ground_truth,
    #     weight1=1.0, weight2=0.0,
    #     model_name="OpenCLIP Only"
    # )
    
    # # 2. BEiT3 only
    # results['BEiT3 Only'] = evaluate_model(
    #     models_dict, ground_truth,
    #     weight1=0.0, weight2=1.0,
    #     model_name="BEiT3 Only"
    # )
    
    # # 3. Fusion (30% + 70%)
    # results['Fusion (30-70)'] = evaluate_model(
    #     models_dict, ground_truth,
    #     weight1=0.3, weight2=0.7,
    #     model_name="Fusion (30% OpenCLIP + 70% BEiT3)"
    # )

    # results['Fusion (50-50)'] = evaluate_model(
    #     models_dict, ground_truth,
    #     weight1=0.5, weight2=0.5,
    #     model_name="Fusion (50% OpenCLIP + 50% BEiT3)"
    # )
    # results['Fusion (70-30)'] = evaluate_model(
    #     models_dict, ground_truth,
    #     weight1=0.7, weight2=0.3,
    #     model_name="Fusion (70% OpenCLIP + 30% BEiT3)"
    # )
    results['Fusion (58-42)'] = evaluate_model(
        models_dict, ground_truth,
        weight1=0.58, weight2=0.42,
        model_name="Fusion (58% OpenCLIP + 42% BEiT3)"
    )
    # Print comparison table
    print("\n" + "="*80)
    print("EVALUATION RESULTS COMPARISON")
    print("="*80)
    print(f"{'Model':<25} {'R@1':>10} {'R@5':>10} {'R@10':>10} {'Mean Rank':>12} {'Median Rank':>12}")
    print("-"*80)
    
    for model_name, metrics in results.items():
        if metrics:
            print(f"{model_name:<25} {metrics['R@1']:>9.2f}% {metrics['R@5']:>9.2f}% "
                  f"{metrics['R@10']:>9.2f}% {metrics['Mean Rank']:>11.2f} {metrics['Median Rank']:>12.1f}")
    
    print("="*80)
    
    # Save results to JSON
    output_file = os.path.join(output_folder, "evaluation_results2.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
