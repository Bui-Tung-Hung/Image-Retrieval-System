"""
SigLIP2 Evaluation Script
Evaluate ViT-B-16-SigLIP2-256 on Vietnamese Flickr8k test set

This script:
1. Encodes 2535 real test images with SigLIP2 Base (256x256, 768 dims)
2. Evaluates retrieval metrics on real Vietnamese captions
3. Benchmarks performance (encoding speed, search latency, memory)
4. Compares with OpenCLIP and BEiT3 results
5. Provides recommendation on whether fine-tuning is needed
"""

import torch
from PIL import Image
import os
import sys
import pickle
import numpy as np
import faiss
import pandas as pd
import json
import time
import random
from tqdm import tqdm
from collections import defaultdict

# Import OpenCLIP
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'open_clip', 'src'))
import open_clip

# ======================
# Configuration - USING REAL DATA
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16  # Smaller due to 384x384 resolution
test_folder = r"D:\Projects\kaggle\test"  # ✅ REAL: 2535 test images
ground_truth_csv = r"D:\Projects\kaggle\test_corrected.csv"  # ✅ REAL: Vietnamese captions
output_folder = r"D:\Projects\doan\siglip2_output"
comparison_results = r"D:\Projects\doan\rank_fusion_output\evaluation_results.json"  # For comparison

# Model configuration
model_name = 'ViT-B-16-SigLIP2-256'     # Base model (400M params)
pretrained = 'webli'
embedding_dim = 768  # Base uses 768 dims (need to verify)
image_size = 256  # Base uses 256x256

# Random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

print(f"Device: {device}")
print(f"Test folder: {test_folder}")
print(f"Ground truth CSV: {ground_truth_csv}")
print(f"Output folder: {output_folder}")
print(f"Batch size: {batch_size}")


# ======================
# Utility Functions
# ======================

class TimingContext:
    """Context manager for precise timing"""
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time


def get_gpu_memory():
    """Get current GPU memory allocated in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def reset_peak_memory():
    """Reset peak memory stats"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory():
    """Get peak GPU memory in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0


def format_time(seconds):
    """Format time with appropriate unit"""
    if seconds < 0.001:
        return f"{seconds*1000000:.2f} μs"
    elif seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"


def format_memory(mb):
    """Format memory with appropriate unit"""
    if mb < 1024:
        return f"{mb:.2f} MB"
    else:
        return f"{mb/1024:.2f} GB"


# ======================
# Model Loading
# ======================

def load_siglip2_model():
    """Load SigLIP2 model and preprocessing"""
    print("\n" + "="*60)
    print("LOADING SIGLIP2 MODEL")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Pretrained: {pretrained}")
    
    try:
        with TimingContext("Model loading") as timer:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained
            )
            model = model.to(device)
            model.eval()
        
        print(f"✓ Model loaded in {format_time(timer.elapsed)}")
        
        # Print model info
        print(f"\nModel Architecture:")
        print(f"  - Vision tower: {model.visual.__class__.__name__}")
        print(f"  - Text tower: {model.text.__class__.__name__}")
        print(f"  - Embedding dimension: {embedding_dim}")
        print(f"  - Image size: {image_size}x{image_size}")
        
        return model, preprocess
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def load_siglip2_tokenizer():
    """Load SigLIP2 tokenizer"""
    print("\nLoading tokenizer...")
    try:
        # Try OpenCLIP's method first
        try:
            tokenizer = open_clip.get_tokenizer(model_name)
            print("✓ Tokenizer loaded via OpenCLIP (Gemma-based, vocab=256k)")
            return tokenizer
        except Exception as e1:
            print(f"⚠ OpenCLIP tokenizer failed: {e1}")
            print("  Trying direct transformers loading...")
            
            # Fallback: Load directly from transformers
            from transformers import AutoTokenizer
            tokenizer_name = 'timm/ViT-SO400M-16-SigLIP2-384'
            
            # Use legacy=False to avoid the parsing error
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                legacy=False,
                trust_remote_code=True
            )
            print("✓ Tokenizer loaded via transformers (Gemma-based, vocab=256k)")
            
            # Wrap in a simple class to match OpenCLIP's interface
            class TokenizerWrapper:
                def __init__(self, hf_tokenizer):
                    self.hf_tokenizer = hf_tokenizer
                
                def __call__(self, texts, context_length=64):
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    # Tokenize and convert to tensor
                    encoded = self.hf_tokenizer(
                        texts,
                        padding='max_length',
                        truncation=True,
                        max_length=context_length,
                        return_tensors='pt'
                    )
                    return encoded['input_ids']
            
            return TokenizerWrapper(tokenizer)
            
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        raise


def test_model_with_dummy():
    """Test model with dummy data to verify it works"""
    print("\nTesting model with dummy data...")
    model, preprocess = load_siglip2_model()
    tokenizer = load_siglip2_tokenizer()
    
    # Test image encoding
    dummy_img = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
    img_tensor = preprocess(dummy_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        img_features = model.encode_image(img_tensor)
    
    print(f"✓ Image encoding test: {img_features.shape} (expected: [1, {embedding_dim}])")
    
    # Test text encoding
    text_tokens = tokenizer(["test caption"]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    
    print(f"✓ Text encoding test: {text_features.shape} (expected: [1, {embedding_dim}])")
    
    print("✓ Model verification successful!")
    return model, preprocess, tokenizer


# ======================
# Image Encoding - USING REAL DATA
# ======================

def scan_test_images(test_folder):
    """Scan and return list of real test images"""
    print("\n" + "="*60)
    print("SCANNING REAL TEST IMAGES")
    print("="*60)
    print(f"Folder: {test_folder}")
    
    if not os.path.exists(test_folder):
        raise FileNotFoundError(f"Test folder not found: {test_folder}")
    
    # Get all image files
    image_list = [
        os.path.join(test_folder, img)
        for img in sorted(os.listdir(test_folder))
        if img.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    
    print(f"✓ Found {len(image_list)} real images")
    
    if len(image_list) == 0:
        raise ValueError("No images found in test folder!")
    
    return image_list


def encode_images_siglip2(model, preprocess, image_list, batch_size=16):
    """Encode all real images using SigLIP2"""
    print("\n" + "="*60)
    print(f"ENCODING {len(image_list)} REAL IMAGES WITH SIGLIP2")
    print("="*60)
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}x{image_size}")
    
    all_embeddings = []
    failed_images = []
    
    with TimingContext("Full image encoding") as total_timer:
        for i in tqdm(range(0, len(image_list), batch_size), desc="Encoding images"):
            batch_paths = image_list[i:i+batch_size]
            
            try:
                # Load and preprocess batch
                batch_images = []
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        batch_images.append(preprocess(img))
                    except Exception as e:
                        print(f"\n⚠ Warning: Failed to load {os.path.basename(path)}: {e}")
                        failed_images.append(path)
                        continue
                
                if len(batch_images) == 0:
                    continue
                
                # Stack and encode
                batch_tensor = torch.stack(batch_images).to(device)
                
                with torch.no_grad():
                    batch_features = model.encode_image(batch_tensor)
                
                all_embeddings.append(batch_features.cpu())
                
            except Exception as e:
                print(f"\n❌ Error processing batch {i//batch_size}: {e}")
                continue
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    print(f"\n✓ Encoding complete!")
    print(f"  - Total time: {format_time(total_timer.elapsed)}")
    print(f"  - Successful: {all_embeddings.shape[0]} images")
    print(f"  - Failed: {len(failed_images)} images")
    print(f"  - Throughput: {all_embeddings.shape[0] / total_timer.elapsed:.2f} images/sec")
    print(f"  - Embedding shape: {all_embeddings.shape} (expected: [{len(image_list)}, {embedding_dim}])")
    
    return all_embeddings, failed_images


# ======================
# FAISS Index Creation
# ======================

def create_faiss_index(embeddings):
    """Create FAISS index from embeddings"""
    print("\n" + "="*60)
    print("CREATING FAISS INDEX")
    print("="*60)
    
    # Convert to numpy
    embeddings_np = embeddings.numpy().astype('float32')
    print(f"Embeddings shape: {embeddings_np.shape}")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_np)
    print("✓ Embeddings normalized")
    
    # Create index
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_np)
    print(f"✓ FAISS index created: {index.ntotal} vectors, dim={embedding_dim}")
    
    return index


def save_outputs(embeddings, index, image_paths):
    """Save embeddings, index, and image paths"""
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Save embeddings
    emb_path = os.path.join(output_folder, "siglip2_embeddings.pt")
    torch.save(embeddings, emb_path)
    print(f"✓ Saved embeddings: {emb_path}")
    
    # Save FAISS index
    index_path = os.path.join(output_folder, "siglip2_image_index.faiss")
    faiss.write_index(index, index_path)
    print(f"✓ Saved FAISS index: {index_path}")
    
    # Save image paths
    paths_file = os.path.join(output_folder, "image_paths.pkl")
    with open(paths_file, "wb") as f:
        pickle.dump(image_paths, f)
    print(f"✓ Saved image paths: {paths_file} ({len(image_paths)} paths)")


# ======================
# Ground Truth Loading - USING REAL DATA
# ======================

def load_ground_truth(csv_path):
    """Load real Vietnamese captions from CSV"""
    print("\n" + "="*60)
    print("LOADING REAL GROUND TRUTH")
    print("="*60)
    print(f"CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path, sep=';', names=['image_filename', 'caption'], encoding='utf-8')
    print(f"✓ Loaded CSV: {len(df)} rows")
    
    # Group captions by image
    gt_dict = defaultdict(list)
    for _, row in df.iterrows():
        image_name = os.path.basename(row['image_filename'])
        gt_dict[image_name].append(row['caption'])
    
    print(f"✓ Grouped captions:")
    print(f"  - Unique images: {len(gt_dict)}")
    print(f"  - Total captions: {len(df)}")
    print(f"  - Avg captions/image: {len(df)/len(gt_dict):.1f}")
    
    # Show sample
    sample_img = list(gt_dict.keys())[0]
    print(f"\nSample (first image):")
    print(f"  Image: {sample_img}")
    for i, cap in enumerate(gt_dict[sample_img][:3], 1):
        print(f"  Caption {i}: {cap[:80]}...")
    
    return gt_dict


def validate_ground_truth(gt_dict, image_list):
    """Validate that ground truth images exist in test folder"""
    print("\n" + "="*60)
    print("VALIDATING GROUND TRUTH")
    print("="*60)
    
    image_names = set(os.path.basename(path) for path in image_list)
    gt_names = set(gt_dict.keys())
    
    # Check matches
    matched = gt_names & image_names
    missing_in_gt = image_names - gt_names
    missing_in_test = gt_names - image_names
    
    print(f"Matched images: {len(matched)}")
    print(f"In test folder but not in GT: {len(missing_in_gt)}")
    print(f"In GT but not in test folder: {len(missing_in_test)}")
    
    if len(missing_in_test) > 0:
        print(f"\n⚠ Warning: {len(missing_in_test)} images in ground truth not found in test folder")
        if len(missing_in_test) <= 10:
            for img in list(missing_in_test)[:10]:
                print(f"  - {img}")
    
    if len(matched) == 0:
        raise ValueError("No matching images between ground truth and test folder!")
    
    print(f"\n✓ Validation complete: {len(matched)} images can be evaluated")
    return matched


# ======================
# Text Encoding - USING REAL VIETNAMESE CAPTIONS
# ======================

def encode_text_siglip2(model, tokenizer, text):
    """Encode single Vietnamese text with SigLIP2"""
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype('float32')


def test_vietnamese_encoding(model, tokenizer):
    """Test encoding real Vietnamese captions"""
    print("\n" + "="*60)
    print("TESTING VIETNAMESE TEXT ENCODING")
    print("="*60)
    
    test_captions = [
        "Một con chó đang chạy trên bãi cỏ",
        "Hai người đang ngồi trên ghế băng",
        "Một cô gái đang cầm ô màu đỏ"
    ]
    
    for i, caption in enumerate(test_captions, 1):
        with TimingContext() as timer:
            features = encode_text_siglip2(model, tokenizer, caption)
        
        print(f"{i}. \"{caption}\"")
        print(f"   Shape: {features.shape}, Norm: {np.linalg.norm(features):.4f}, Time: {format_time(timer.elapsed)}")
    
    print("✓ Vietnamese encoding test successful!")


# ======================
# Retrieval Search
# ======================

def search_siglip2(query_text, model, tokenizer, index, image_paths, top_k=10):
    """Search for images using Vietnamese text query"""
    # Encode query
    query_emb = encode_text_siglip2(model, tokenizer, query_text)
    
    # Normalize
    faiss.normalize_L2(query_emb)
    
    # Search
    k = min(top_k, index.ntotal)
    similarities, indices = index.search(query_emb, k)
    
    # Format results
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], similarities[0]), 1):
        results.append({
            'rank': rank,
            'image_path': image_paths[idx],
            'image_name': os.path.basename(image_paths[idx]),
            'score': float(score)
        })
    
    return results


# ======================
# Metrics Calculation
# ======================

def calculate_retrieval_metrics(ground_truth, all_results, image_paths):
    """Calculate R@1, R@5, R@10, Mean/Median Rank from real queries"""
    print("\n" + "="*60)
    print("CALCULATING RETRIEVAL METRICS")
    print("="*60)
    
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
        for result in search_results:
            if result['image_name'] == gt_image:
                rank = result['rank']
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
        'Mean Rank': float(np.mean(ranks)),
        'Median Rank': float(np.median(ranks)),
        'Total Queries': total_queries
    }
    
    print(f"Total queries evaluated: {total_queries}")
    print(f"R@1:  {metrics['R@1']:.2f}%")
    print(f"R@5:  {metrics['R@5']:.2f}%")
    print(f"R@10: {metrics['R@10']:.2f}%")
    print(f"Mean Rank: {metrics['Mean Rank']:.2f}")
    print(f"Median Rank: {metrics['Median Rank']:.1f}")
    
    return metrics


# ======================
# Performance Benchmarking
# ======================

def benchmark_image_encoding(model, preprocess, image_list):
    """Benchmark image encoding performance"""
    print("\n" + "="*60)
    print("BENCHMARKING IMAGE ENCODING")
    print("="*60)
    
    results = {}
    
    # Single image encoding
    print("\n1. Single image encoding (100 iterations)...")
    sample_img = Image.open(image_list[0]).convert("RGB")
    sample_tensor = preprocess(sample_img).unsqueeze(0).to(device)
    
    times = []
    for _ in range(100):
        with TimingContext() as timer:
            with torch.no_grad():
                _ = model.encode_image(sample_tensor)
        times.append(timer.elapsed)
    
    results['single_image'] = {
        'avg_time': np.mean(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'throughput': 1.0 / np.mean(times)
    }
    print(f"   Avg: {format_time(np.mean(times))}")
    print(f"   Throughput: {results['single_image']['throughput']:.2f} images/sec")
    
    # Batch encoding
    print("\n2. Batch encoding (batch_size={}, 20 iterations)...".format(batch_size))
    sample_images = [Image.open(img).convert("RGB") for img in image_list[:batch_size]]
    sample_batch = torch.stack([preprocess(img) for img in sample_images]).to(device)
    
    times = []
    for _ in range(20):
        with TimingContext() as timer:
            with torch.no_grad():
                _ = model.encode_image(sample_batch)
        times.append(timer.elapsed)
    
    results['batch_encoding'] = {
        'avg_time': np.mean(times),
        'per_image_time': np.mean(times) / batch_size,
        'throughput': batch_size / np.mean(times)
    }
    print(f"   Avg batch time: {format_time(np.mean(times))}")
    print(f"   Per image: {format_time(results['batch_encoding']['per_image_time'])}")
    print(f"   Throughput: {results['batch_encoding']['throughput']:.2f} images/sec")
    
    return results


def benchmark_text_encoding(model, tokenizer, ground_truth):
    """Benchmark text encoding performance with real Vietnamese captions"""
    print("\n" + "="*60)
    print("BENCHMARKING TEXT ENCODING")
    print("="*60)
    
    results = {}
    
    # Get sample captions
    all_captions = []
    for captions in ground_truth.values():
        all_captions.extend(captions)
    
    # Single text encoding
    print("\n1. Single text encoding (100 iterations)...")
    sample_caption = all_captions[0]
    
    times = []
    for _ in range(100):
        with TimingContext() as timer:
            _ = encode_text_siglip2(model, tokenizer, sample_caption)
        times.append(timer.elapsed)
    
    results['single_text'] = {
        'avg_time': np.mean(times),
        'throughput': 1.0 / np.mean(times)
    }
    print(f"   Avg: {format_time(np.mean(times))}")
    print(f"   Throughput: {results['single_text']['throughput']:.2f} queries/sec")
    
    return results


def benchmark_search_latency(model, tokenizer, index, image_paths, ground_truth):
    """Benchmark search latency for different top_k values"""
    print("\n" + "="*60)
    print("BENCHMARKING SEARCH LATENCY")
    print("="*60)
    
    # Get sample queries
    all_queries = []
    for captions in ground_truth.values():
        all_queries.extend(captions)
    
    sample_queries = random.sample(all_queries, min(50, len(all_queries)))
    
    results = {}
    top_k_values = [1, 5, 10, 50, 100]
    
    for k in top_k_values:
        print(f"\nTesting top_k={k} ({len(sample_queries)} queries)...")
        times = []
        
        for query in sample_queries:
            with TimingContext() as timer:
                _ = search_siglip2(query, model, tokenizer, index, image_paths, top_k=k)
            times.append(timer.elapsed)
        
        results[f'top_{k}'] = {
            'avg_latency': np.mean(times),
            'min_latency': np.min(times),
            'max_latency': np.max(times),
            'throughput': 1.0 / np.mean(times)
        }
        print(f"   Avg latency: {format_time(np.mean(times))}")
        print(f"   Throughput: {results[f'top_{k}']['throughput']:.2f} queries/sec")
    
    return results


def benchmark_gpu_memory():
    """Measure GPU memory usage"""
    print("\n" + "="*60)
    print("GPU MEMORY USAGE")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return {}
    
    results = {
        'current_allocated': get_gpu_memory(),
        'peak_allocated': get_peak_memory()
    }
    
    print(f"Current allocated: {format_memory(results['current_allocated'])}")
    print(f"Peak allocated: {format_memory(results['peak_allocated'])}")
    
    return results


# ======================
# Comparison with Existing Models
# ======================

def load_existing_results():
    """Load existing evaluation results for comparison"""
    print("\n" + "="*60)
    print("LOADING EXISTING RESULTS FOR COMPARISON")
    print("="*60)
    
    if not os.path.exists(comparison_results):
        print(f"⚠ Comparison file not found: {comparison_results}")
        print("  Skipping comparison")
        return None
    
    try:
        with open(comparison_results, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"✓ Loaded results for {len(results)} configurations:")
        for model_name in results.keys():
            print(f"  - {model_name}")
        
        return results
    except Exception as e:
        print(f"❌ Error loading comparison results: {e}")
        return None


def generate_comparison_table(siglip2_metrics, existing_results):
    """Generate comparison table"""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<30} {'R@1':>10} {'R@5':>10} {'R@10':>10} {'Mean Rank':>12} {'Median Rank':>12}")
    print("-"*80)
    
    # Print existing results
    if existing_results:
        for model_name, metrics in existing_results.items():
            if metrics and 'R@1' in metrics:
                print(f"{model_name:<30} {metrics['R@1']:>9.2f}% {metrics['R@5']:>9.2f}% "
                      f"{metrics['R@10']:>9.2f}% {metrics['Mean Rank']:>11.2f} {metrics['Median Rank']:>12.1f}")
    
    # Print SigLIP2 results
    if siglip2_metrics:
        print(f"{'SigLIP2 (pretrained webli)':<30} {siglip2_metrics['R@1']:>9.2f}% {siglip2_metrics['R@5']:>9.2f}% "
              f"{siglip2_metrics['R@10']:>9.2f}% {siglip2_metrics['Mean Rank']:>11.2f} {siglip2_metrics['Median Rank']:>12.1f}")
    
    print("="*80)


def generate_recommendation(siglip2_metrics, existing_results):
    """Generate recommendation on fine-tuning"""
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    r1 = siglip2_metrics['R@1']
    
    # Compare with existing models if available
    if existing_results:
        best_existing_r1 = max(
            metrics.get('R@1', 0) 
            for metrics in existing_results.values() 
            if metrics
        )
        
        print(f"\nSigLIP2 R@1: {r1:.2f}%")
        print(f"Best existing R@1: {best_existing_r1:.2f}%")
        print(f"Difference: {r1 - best_existing_r1:+.2f}%")
        
        if r1 > best_existing_r1:
            print("\n✅ VERDICT: SigLIP2 outperforms existing models!")
            print("   Recommendation: Use SigLIP2 pretrained (no fine-tuning needed)")
        elif r1 > best_existing_r1 * 0.9:
            print("\n⚡ VERDICT: SigLIP2 performance is competitive")
            print("   Recommendation: Fine-tuning may provide marginal improvement")
        else:
            print("\n⚠ VERDICT: SigLIP2 underperforms compared to existing models")
            print("   Recommendation: Fine-tuning on Vietnamese data is STRONGLY RECOMMENDED")
    else:
        # No comparison available, use absolute thresholds
        if r1 > 40:
            print(f"\n✅ VERDICT: Excellent performance (R@1 = {r1:.2f}%)")
            print("   Recommendation: Use pretrained, fine-tuning optional for further gains")
        elif r1 > 25:
            print(f"\n⚡ VERDICT: Good performance (R@1 = {r1:.2f}%)")
            print("   Recommendation: Fine-tuning may improve to excellent levels")
        elif r1 > 15:
            print(f"\n⚠ VERDICT: Moderate performance (R@1 = {r1:.2f}%)")
            print("   Recommendation: Fine-tuning RECOMMENDED")
        else:
            print(f"\n❌ VERDICT: Poor performance (R@1 = {r1:.2f}%)")
            print("   Recommendation: Fine-tuning STRONGLY RECOMMENDED")


# ======================
# Main Evaluation Pipeline
# ======================

def run_full_evaluation():
    """Run complete SigLIP2 evaluation on real data"""
    print("="*80)
    print(" "*25 + "SIGLIP2 EVALUATION")
    print(" "*15 + "Vietnamese Flickr8k Test Set (REAL DATA)")
    print("="*80)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Phase 1: Load model
    model, preprocess = load_siglip2_model()
    tokenizer = load_siglip2_tokenizer()
    test_vietnamese_encoding(model, tokenizer)
    
    # Phase 2: Encode all real images
    image_list = scan_test_images(test_folder)
    embeddings, failed = encode_images_siglip2(model, preprocess, image_list, batch_size)
    
    # Remove failed images from list
    if failed:
        image_list = [img for img in image_list if img not in failed]
    
    # Phase 3: Create FAISS index
    index = create_faiss_index(embeddings)
    save_outputs(embeddings, index, image_list)
    
    # Phase 4: Load real ground truth
    ground_truth = load_ground_truth(ground_truth_csv)
    matched_images = validate_ground_truth(ground_truth, image_list)
    
    # Phase 5: Run retrieval evaluation
    print("\n" + "="*60)
    print("RUNNING RETRIEVAL EVALUATION")
    print("="*60)
    
    all_results = []
    all_queries = []
    for img_name, captions in ground_truth.items():
        all_queries.extend(captions)
    
    print(f"Evaluating {len(all_queries)} real Vietnamese queries...")
    
    for query in tqdm(all_queries, desc="Searching"):
        results = search_siglip2(query, model, tokenizer, index, image_list, top_k=10)
        all_results.append((query, results))
    
    metrics = calculate_retrieval_metrics(ground_truth, all_results, image_list)
    
    # Phase 6: Performance benchmarking
    perf_img = benchmark_image_encoding(model, preprocess, image_list)
    perf_text = benchmark_text_encoding(model, tokenizer, ground_truth)
    perf_search = benchmark_search_latency(model, tokenizer, index, image_list, ground_truth)
    perf_memory = benchmark_gpu_memory()
    
    # Phase 7: Comparison and recommendation
    existing_results = load_existing_results()
    generate_comparison_table(metrics, existing_results)
    generate_recommendation(metrics, existing_results)
    
    # Phase 8: Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    final_results = {
        'model': model_name,
        'pretrained': pretrained,
        'metrics': metrics,
        'performance': {
            'image_encoding': perf_img,
            'text_encoding': perf_text,
            'search': perf_search,
            'memory': perf_memory
        },
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    results_file = os.path.join(output_folder, "siglip2_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to: {results_file}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nOutput files in: {output_folder}")
    print("  - siglip2_embeddings.pt")
    print("  - siglip2_image_index.faiss")
    print("  - image_paths.pkl")
    print("  - siglip2_results.json")
    print("="*80)
    
    return final_results


def main():
    """Main entry point"""
    try:
        results = run_full_evaluation()
        return results
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return None
    except Exception as e:
        print(f"\n\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
