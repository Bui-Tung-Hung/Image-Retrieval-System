import torch
import os
import sys
import pickle
import numpy as np
import faiss
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
openclip_ckpt = r"D:\Projects\doan\open_clip\checkpoints\epoch_15.pt"
beit3_ckpt = r"C:\Users\LAPTOP\Downloads\BEiT3\ckpt\checkpoint-best.pth"
beit3_spm = r"D:\Projects\doan\beit3\beit3.spm"


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
    print(f"✓ Loaded FAISS indices")
    
    # Load OpenCLIP model
    print("Loading OpenCLIP model...")
    openclip_model, _, _ = open_clip.create_model_and_transforms(
        model_name='ViT-B-16',
        pretrained=openclip_ckpt
    )
    openclip_model = openclip_model.to(device).eval()
    openclip_tokenizer = open_clip.get_tokenizer('ViT-B-16')
    print("✓ OpenCLIP ready")
    
    # Load BEiT3 model
    print("Loading BEiT3 model...")
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
    
    print("="*60)
    print("✅ All models loaded successfully!")
    print("="*60)
    
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


def search_single_model(query_text, model, tokenizer, index, image_paths, model_type='openclip', top_k=5):
    """Search using a single model"""
    if model_type == 'openclip':
        query_emb = encode_text_openclip(model, tokenizer, query_text)
    else:
        query_emb = encode_text_beit3(model, tokenizer, query_text)
    
    faiss.normalize_L2(query_emb)
    sims, indices = index.search(query_emb, top_k)
    
    results = []
    for i, (idx, score) in enumerate(zip(indices[0], sims[0])):
        results.append({
            'rank': i + 1,
            'image_path': image_paths[idx],
            'image_name': os.path.basename(image_paths[idx]),
            'score': float(score)
        })
    
    return results


def search_with_fusion(query_text, models_dict, weight1=0.3, weight2=0.7, top_k=5):
    """Search with rank fusion"""
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
    k = min(top_k * 10, models_dict['index_openclip'].ntotal)
    
    sims1, indices1 = models_dict['index_openclip'].search(query_emb1, k)
    sims2, indices2 = models_dict['index_beit3'].search(query_emb2, k)
    
    # Fusion
    fusion_scores = {}
    for i, idx in enumerate(indices1[0]):
        fusion_scores[idx] = weight1 * sims1[0][i]
    
    for i, idx in enumerate(indices2[0]):
        if idx in fusion_scores:
            fusion_scores[idx] += weight2 * sims2[0][i]
        else:
            fusion_scores[idx] = weight2 * sims2[0][i]
    
    # Sort and get top-k
    sorted_results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for rank, (idx, score) in enumerate(sorted_results, 1):
        results.append({
            'rank': rank,
            'image_path': models_dict['image_paths'][idx],
            'image_name': os.path.basename(models_dict['image_paths'][idx]),
            'score': float(score)
        })
    
    return results


def display_results(results, title="Search Results"):
    """Display search results in a nice format"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Image Name':<50} {'Score':>10}")
    print("-"*80)
    
    for result in results:
        print(f"{result['rank']:<6} {result['image_name']:<50} {result['score']:>10.4f}")
    
    print("="*80)


def compare_all_modes(query_text, models_dict, top_k=5):
    """Compare results from all three modes side by side"""
    print(f"\n{'='*80}")
    print(f"COMPARISON MODE - Query: '{query_text}'")
    print(f"{'='*80}")
    
    # OpenCLIP only
    print("\n[1] OpenCLIP Only:")
    results_openclip = search_single_model(
        query_text,
        models_dict['openclip_model'],
        models_dict['openclip_tokenizer'],
        models_dict['index_openclip'],
        models_dict['image_paths'],
        model_type='openclip',
        top_k=top_k
    )
    display_results(results_openclip, "OpenCLIP Results")
    
    # BEiT3 only
    print("\n[2] BEiT3 Only:")
    results_beit3 = search_single_model(
        query_text,
        models_dict['beit3_model'],
        models_dict['beit3_tokenizer'],
        models_dict['index_beit3'],
        models_dict['image_paths'],
        model_type='beit3',
        top_k=top_k
    )
    display_results(results_beit3, "BEiT3 Results")
    
    # Fusion
    print("\n[3] Fusion (30% OpenCLIP + 70% BEiT3):")
    results_fusion = search_with_fusion(query_text, models_dict, weight1=0.3, weight2=0.7, top_k=top_k)
    display_results(results_fusion, "Fusion Results")


def main():
    print("="*80)
    print(" "*25 + "RANK FUSION DEMO")
    print(" "*15 + "Interactive Image Search with Model Fusion")
    print("="*80)
    
    # Load models
    models_dict = load_models_and_indices()
    
    print("\n" + "="*80)
    print("COMMANDS:")
    print("  - Type your query and press Enter to search")
    print("  - Type 'compare' to see results from all models side-by-side")
    print("  - Type 'quit' or 'exit' to quit")
    print("="*80)
    
    while True:
        try:
            print("\n" + "-"*80)
            query = input("Enter your query (or 'quit' to exit): ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'compare':
                compare_query = input("Enter query for comparison: ").strip()
                if compare_query:
                    compare_all_modes(compare_query, models_dict, top_k=5)
                continue
            
            # Default: show fusion results
            print(f"\n🔍 Searching for: '{query}'")
            results = search_with_fusion(query, models_dict, weight1=0.3, weight2=0.7, top_k=5)
            display_results(results, f"Fusion Results (30% OpenCLIP + 70% BEiT3)")
            
            # Ask if user wants to see comparison
            show_compare = input("\nShow comparison with individual models? (y/n): ").strip().lower()
            if show_compare == 'y':
                compare_all_modes(query, models_dict, top_k=5)
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
