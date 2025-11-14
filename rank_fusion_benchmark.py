import torch
from PIL import Image
import os
import sys
import pickle
import numpy as np
import faiss
import pandas as pd
import time
import random
from tqdm import tqdm
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model
from transformers import XLMRobertaTokenizer
from collections import defaultdict

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
batch_size = 32
test_folder = r"D:\Projects\kaggle\test"
ground_truth_csv = r"D:\Projects\kaggle\test_corrected.csv"
output_folder = r"D:\Projects\doan\rank_fusion_output"
openclip_ckpt = r"D:\Projects\doan\open_clip\checkpoints\epoch_15.pt"
beit3_ckpt = r"C:\Users\LAPTOP\Downloads\BEiT3\ckpt\checkpoint-best.pth"
beit3_spm = r"D:\Projects\doan\beit3\beit3.spm"

# Random seed for reproducibility
random.seed(42)


# ======================
# Utility Functions
# ======================

class TimingContext:
    """Context manager for timing code blocks"""
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


def print_section_header(title):
    """Print a section header with decorations"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)


def print_table(headers, rows, title=None):
    """Print a formatted ASCII table"""
    if title:
        print_section_header(title)
    else:
        print("\n" + "-"*80)
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print header
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in rows:
        row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(row_line)
    
    print("-"*80)


def calculate_percentage_diff(val1, val2):
    """Calculate percentage difference"""
    if val2 == 0:
        return "N/A"
    diff = ((val1 - val2) / val2) * 100
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.1f}%"


# ======================
# Model Loading
# ======================

def load_models_and_data():
    """Load all models, indices, and data"""
    print_section_header("LOADING MODELS AND DATA")
    
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
    openclip_model, _, openclip_preprocess = open_clip.create_model_and_transforms(
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
    
    # BEiT3 preprocess
    beit3_preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])
    print("✓ BEiT3 ready")
    
    # Load ground truth for sample queries
    print("Loading ground truth...")
    df = pd.read_csv(ground_truth_csv, sep=';', names=['image_filename', 'caption'], encoding='utf-8')
    all_captions = df['caption'].tolist()
    sample_queries = random.sample(all_captions, min(100, len(all_captions)))
    print(f"✓ Sampled {len(sample_queries)} queries")
    
    print("="*80)
    print("✅ All models and data loaded!\n")
    
    return {
        'openclip_model': openclip_model,
        'openclip_tokenizer': openclip_tokenizer,
        'openclip_preprocess': openclip_preprocess,
        'beit3_model': beit3_model,
        'beit3_tokenizer': beit3_tokenizer,
        'beit3_preprocess': beit3_preprocess,
        'index_openclip': index_openclip,
        'index_beit3': index_beit3,
        'image_paths': image_paths,
        'sample_queries': sample_queries
    }


# ======================
# Image Encoding Benchmarks
# ======================

def encode_image_openclip(model, preprocess, image_path):
    """Encode single image with OpenCLIP"""
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_tensor)
        features /= features.norm(dim=-1, keepdim=True)
    return features


def encode_image_beit3(model, preprocess, image_path):
    """Encode single image with BEiT3"""
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vision_cls, _ = model(image=img_tensor, only_infer=True)
    return vision_cls


def benchmark_single_image_encoding(model, preprocess, image_path, model_name, encode_func):
    """Benchmark encoding of a single image"""
    reset_peak_memory()
    
    with TimingContext() as timer:
        _ = encode_func(model, preprocess, image_path)
    
    peak_mem = get_peak_memory()
    
    return {
        'time': timer.elapsed,
        'vram': peak_mem
    }


def benchmark_batch_encoding(model, preprocess, image_paths, model_name, encode_func_name):
    """Benchmark encoding of a batch of images"""
    reset_peak_memory()
    
    batch_images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        batch_images.append(preprocess(img))
    
    batch_tensor = torch.stack(batch_images).to(device)
    
    with TimingContext() as timer:
        with torch.no_grad():
            if encode_func_name == 'openclip':
                features = model.encode_image(batch_tensor)
                features /= features.norm(dim=-1, keepdim=True)
            else:  # beit3
                vision_cls, _ = model(image=batch_tensor, only_infer=True)
    
    peak_mem = get_peak_memory()
    
    return {
        'time': timer.elapsed,
        'vram': peak_mem,
        'per_image': timer.elapsed / len(image_paths)
    }


def benchmark_full_encoding(model, preprocess, all_image_paths, model_name, encode_func_name):
    """Benchmark encoding of all images"""
    reset_peak_memory()
    
    total_images = len(all_image_paths)
    
    with TimingContext() as timer:
        for i in tqdm(range(0, total_images, batch_size), desc=f"{model_name} full encoding"):
            batch_paths = all_image_paths[i:i+batch_size]
            batch_images = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                batch_images.append(preprocess(img))
            
            batch_tensor = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                if encode_func_name == 'openclip':
                    features = model.encode_image(batch_tensor)
                    features /= features.norm(dim=-1, keepdim=True)
                else:  # beit3
                    vision_cls, _ = model(image=batch_tensor, only_infer=True)
    
    peak_mem = get_peak_memory()
    throughput = total_images / timer.elapsed
    
    return {
        'time': timer.elapsed,
        'vram': peak_mem,
        'throughput': throughput
    }


def run_image_encoding_benchmarks(data_dict):
    """Run all image encoding benchmarks"""
    print_section_header("IMAGE ENCODING PERFORMANCE BENCHMARKS")
    
    image_paths = data_dict['image_paths']
    sample_image = image_paths[0]
    batch_images = image_paths[:batch_size]
    
    results = {}
    
    # OpenCLIP benchmarks
    print("\n[1/2] Benchmarking OpenCLIP image encoding...")
    results['openclip'] = {}
    
    single_result = benchmark_single_image_encoding(
        data_dict['openclip_model'],
        data_dict['openclip_preprocess'],
        sample_image,
        'OpenCLIP',
        encode_image_openclip
    )
    results['openclip']['single'] = single_result
    print(f"  Single image: {format_time(single_result['time'])}, VRAM: {format_memory(single_result['vram'])}")
    
    batch_result = benchmark_batch_encoding(
        data_dict['openclip_model'],
        data_dict['openclip_preprocess'],
        batch_images,
        'OpenCLIP',
        'openclip'
    )
    results['openclip']['batch'] = batch_result
    print(f"  Batch ({batch_size}): {format_time(batch_result['time'])}, per image: {format_time(batch_result['per_image'])}")
    
    full_result = benchmark_full_encoding(
        data_dict['openclip_model'],
        data_dict['openclip_preprocess'],
        image_paths,
        'OpenCLIP',
        'openclip'
    )
    results['openclip']['full'] = full_result
    print(f"  Full ({len(image_paths)}): {format_time(full_result['time'])}, throughput: {full_result['throughput']:.2f} img/s")
    
    # BEiT3 benchmarks
    print("\n[2/2] Benchmarking BEiT3 image encoding...")
    results['beit3'] = {}
    
    single_result = benchmark_single_image_encoding(
        data_dict['beit3_model'],
        data_dict['beit3_preprocess'],
        sample_image,
        'BEiT3',
        encode_image_beit3
    )
    results['beit3']['single'] = single_result
    print(f"  Single image: {format_time(single_result['time'])}, VRAM: {format_memory(single_result['vram'])}")
    
    batch_result = benchmark_batch_encoding(
        data_dict['beit3_model'],
        data_dict['beit3_preprocess'],
        batch_images,
        'BEiT3',
        'beit3'
    )
    results['beit3']['batch'] = batch_result
    print(f"  Batch ({batch_size}): {format_time(batch_result['time'])}, per image: {format_time(batch_result['per_image'])}")
    
    full_result = benchmark_full_encoding(
        data_dict['beit3_model'],
        data_dict['beit3_preprocess'],
        image_paths,
        'BEiT3',
        'beit3'
    )
    results['beit3']['full'] = full_result
    print(f"  Full ({len(image_paths)}): {format_time(full_result['time'])}, throughput: {full_result['throughput']:.2f} img/s")
    
    # Comparison table
    headers = ["Metric", "OpenCLIP", "BEiT3", "Difference"]
    rows = [
        ["Single Image", 
         format_time(results['openclip']['single']['time']),
         format_time(results['beit3']['single']['time']),
         calculate_percentage_diff(results['beit3']['single']['time'], results['openclip']['single']['time'])],
        ["Batch (32 imgs)",
         format_time(results['openclip']['batch']['time']),
         format_time(results['beit3']['batch']['time']),
         calculate_percentage_diff(results['beit3']['batch']['time'], results['openclip']['batch']['time'])],
        ["Full (2535 imgs)",
         format_time(results['openclip']['full']['time']),
         format_time(results['beit3']['full']['time']),
         calculate_percentage_diff(results['beit3']['full']['time'], results['openclip']['full']['time'])],
        ["Throughput",
         f"{results['openclip']['full']['throughput']:.2f} img/s",
         f"{results['beit3']['full']['throughput']:.2f} img/s",
         calculate_percentage_diff(results['beit3']['full']['throughput'], results['openclip']['full']['throughput'])],
        ["Peak VRAM",
         format_memory(results['openclip']['full']['vram']),
         format_memory(results['beit3']['full']['vram']),
         calculate_percentage_diff(results['beit3']['full']['vram'], results['openclip']['full']['vram'])]
    ]
    
    print_table(headers, rows, "IMAGE ENCODING COMPARISON")
    
    return results


# ======================
# Text Encoding Benchmarks
# ======================

def encode_text_openclip(model, tokenizer, text):
    """Encode text with OpenCLIP"""
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    return features


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
    
    return language_cls


def benchmark_single_text_encoding(model, tokenizer, query, model_name, encode_func):
    """Benchmark encoding single query"""
    with TimingContext() as timer:
        _ = encode_func(model, tokenizer, query)
    
    return {'time': timer.elapsed}


def benchmark_batch_text_encoding(model, tokenizer, queries, model_name, encode_func):
    """Benchmark encoding batch of queries"""
    with TimingContext() as timer:
        for query in queries:
            _ = encode_func(model, tokenizer, query)
    
    return {
        'time': timer.elapsed,
        'per_query': timer.elapsed / len(queries)
    }


def run_text_encoding_benchmarks(data_dict):
    """Run all text encoding benchmarks"""
    print_section_header("TEXT ENCODING PERFORMANCE BENCHMARKS")
    
    sample_queries = data_dict['sample_queries']
    single_query = sample_queries[0]
    batch_queries = sample_queries[:100]
    
    results = {}
    
    # OpenCLIP benchmarks
    print("\n[1/2] Benchmarking OpenCLIP text encoding...")
    results['openclip'] = {}
    
    single_result = benchmark_single_text_encoding(
        data_dict['openclip_model'],
        data_dict['openclip_tokenizer'],
        single_query,
        'OpenCLIP',
        encode_text_openclip
    )
    results['openclip']['single'] = single_result
    print(f"  Single query: {format_time(single_result['time'])}")
    
    batch_result = benchmark_batch_text_encoding(
        data_dict['openclip_model'],
        data_dict['openclip_tokenizer'],
        batch_queries,
        'OpenCLIP',
        encode_text_openclip
    )
    results['openclip']['batch'] = batch_result
    print(f"  Batch ({len(batch_queries)}): {format_time(batch_result['time'])}, per query: {format_time(batch_result['per_query'])}")
    
    # BEiT3 benchmarks
    print("\n[2/2] Benchmarking BEiT3 text encoding...")
    results['beit3'] = {}
    
    single_result = benchmark_single_text_encoding(
        data_dict['beit3_model'],
        data_dict['beit3_tokenizer'],
        single_query,
        'BEiT3',
        encode_text_beit3
    )
    results['beit3']['single'] = single_result
    print(f"  Single query: {format_time(single_result['time'])}")
    
    batch_result = benchmark_batch_text_encoding(
        data_dict['beit3_model'],
        data_dict['beit3_tokenizer'],
        batch_queries,
        'BEiT3',
        encode_text_beit3
    )
    results['beit3']['batch'] = batch_result
    print(f"  Batch ({len(batch_queries)}): {format_time(batch_result['time'])}, per query: {format_time(batch_result['per_query'])}")
    
    # Comparison table
    headers = ["Metric", "OpenCLIP", "BEiT3", "Difference"]
    rows = [
        ["Single Query",
         format_time(results['openclip']['single']['time']),
         format_time(results['beit3']['single']['time']),
         calculate_percentage_diff(results['beit3']['single']['time'], results['openclip']['single']['time'])],
        ["Batch (100 queries)",
         format_time(results['openclip']['batch']['time']),
         format_time(results['beit3']['batch']['time']),
         calculate_percentage_diff(results['beit3']['batch']['time'], results['openclip']['batch']['time'])],
        ["Per Query (avg)",
         format_time(results['openclip']['batch']['per_query']),
         format_time(results['beit3']['batch']['per_query']),
         calculate_percentage_diff(results['beit3']['batch']['per_query'], results['openclip']['batch']['per_query'])],
        ["Throughput",
         f"{1/results['openclip']['batch']['per_query']:.2f} q/s",
         f"{1/results['beit3']['batch']['per_query']:.2f} q/s",
         calculate_percentage_diff(1/results['beit3']['batch']['per_query'], 1/results['openclip']['batch']['per_query'])]
    ]
    
    print_table(headers, rows, "TEXT ENCODING COMPARISON")
    
    return results


# ======================
# Search Benchmarks
# ======================

def benchmark_search_single_model(query, model, tokenizer, index, model_type, top_k, encode_func):
    """Benchmark search with single model"""
    # Encode query
    with TimingContext() as encode_timer:
        query_emb = encode_func(model, tokenizer, query)
        query_np = query_emb.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_np)
    
    # FAISS search
    with TimingContext() as search_timer:
        sims, indices = index.search(query_np, top_k)
    
    total_time = encode_timer.elapsed + search_timer.elapsed
    
    return {
        'encode_time': encode_timer.elapsed,
        'search_time': search_timer.elapsed,
        'total_time': total_time
    }


def benchmark_search_fusion(query, data_dict, top_k, weight1=0.3, weight2=0.7):
    """Benchmark search with fusion"""
    # Encode with OpenCLIP
    with TimingContext() as encode1_timer:
        query_emb1 = encode_text_openclip(
            data_dict['openclip_model'],
            data_dict['openclip_tokenizer'],
            query
        )
        query_np1 = query_emb1.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_np1)
    
    # Encode with BEiT3
    with TimingContext() as encode2_timer:
        query_emb2 = encode_text_beit3(
            data_dict['beit3_model'],
            data_dict['beit3_tokenizer'],
            query
        )
        query_np2 = query_emb2.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_np2)
    
    # Search both indices
    k = min(top_k * 10, data_dict['index_openclip'].ntotal)
    
    with TimingContext() as search1_timer:
        sims1, indices1 = data_dict['index_openclip'].search(query_np1, k)
    
    with TimingContext() as search2_timer:
        sims2, indices2 = data_dict['index_beit3'].search(query_np2, k)
    
    # Fusion
    with TimingContext() as fusion_timer:
        fusion_scores = {}
        for i, idx in enumerate(indices1[0]):
            fusion_scores[idx] = weight1 * sims1[0][i]
        for i, idx in enumerate(indices2[0]):
            if idx in fusion_scores:
                fusion_scores[idx] += weight2 * sims2[0][i]
            else:
                fusion_scores[idx] = weight2 * sims2[0][i]
        sorted_results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    total_time = (encode1_timer.elapsed + encode2_timer.elapsed + 
                  search1_timer.elapsed + search2_timer.elapsed + 
                  fusion_timer.elapsed)
    
    return {
        'encode1_time': encode1_timer.elapsed,
        'encode2_time': encode2_timer.elapsed,
        'search1_time': search1_timer.elapsed,
        'search2_time': search2_timer.elapsed,
        'fusion_time': fusion_timer.elapsed,
        'total_time': total_time
    }


def run_search_benchmarks(data_dict, top_k_values=[1, 5, 10, 50, 100]):
    """Run search benchmarks with different top-k values"""
    print_section_header("SEARCH PERFORMANCE BENCHMARKS (VARYING TOP-K)")
    
    sample_queries = data_dict['sample_queries'][:10]  # Use 10 queries for detailed benchmark
    
    results = {k: {'openclip': [], 'beit3': [], 'fusion': []} for k in top_k_values}
    
    for top_k in top_k_values:
        print(f"\n{'='*60}")
        print(f"Benchmarking with top_k = {top_k}")
        print(f"{'='*60}")
        
        for query in tqdm(sample_queries, desc=f"k={top_k}"):
            # OpenCLIP
            oc_result = benchmark_search_single_model(
                query,
                data_dict['openclip_model'],
                data_dict['openclip_tokenizer'],
                data_dict['index_openclip'],
                'openclip',
                top_k,
                encode_text_openclip
            )
            results[top_k]['openclip'].append(oc_result)
            
            # BEiT3
            b3_result = benchmark_search_single_model(
                query,
                data_dict['beit3_model'],
                data_dict['beit3_tokenizer'],
                data_dict['index_beit3'],
                'beit3',
                top_k,
                encode_text_beit3
            )
            results[top_k]['beit3'].append(b3_result)
            
            # Fusion
            fusion_result = benchmark_search_fusion(query, data_dict, top_k)
            results[top_k]['fusion'].append(fusion_result)
    
    # Calculate averages and print comparison table
    headers = ["Model", "k=1", "k=5", "k=10", "k=50", "k=100"]
    rows = []
    
    for model_name in ['openclip', 'beit3', 'fusion']:
        row = [model_name.upper()]
        for k in top_k_values:
            avg_time = np.mean([r['total_time'] for r in results[k][model_name]])
            row.append(format_time(avg_time))
        rows.append(row)
    
    # Add overhead row
    overhead_row = ["Fusion Overhead"]
    for k in top_k_values:
        fusion_avg = np.mean([r['total_time'] for r in results[k]['fusion']])
        openclip_avg = np.mean([r['total_time'] for r in results[k]['openclip']])
        overhead = calculate_percentage_diff(fusion_avg, openclip_avg)
        overhead_row.append(overhead)
    rows.append(overhead_row)
    
    print_table(headers, rows, "SEARCH LATENCY BY TOP-K (Average over 10 queries)")
    
    # Detailed breakdown for k=10
    print_section_header("DETAILED LATENCY BREAKDOWN (k=10)")
    k = 10
    
    oc_avg = {key: np.mean([r[key] for r in results[k]['openclip']]) for key in results[k]['openclip'][0].keys()}
    b3_avg = {key: np.mean([r[key] for r in results[k]['beit3']]) for key in results[k]['beit3'][0].keys()}
    fusion_avg = {key: np.mean([r[key] for r in results[k]['fusion']]) for key in results[k]['fusion'][0].keys()}
    
    headers = ["Component", "OpenCLIP", "BEiT3", "Fusion"]
    rows = [
        ["Text Encoding", 
         format_time(oc_avg['encode_time']),
         format_time(b3_avg['encode_time']),
         format_time(fusion_avg['encode1_time'] + fusion_avg['encode2_time'])],
        ["FAISS Search",
         format_time(oc_avg['search_time']),
         format_time(b3_avg['search_time']),
         format_time(fusion_avg['search1_time'] + fusion_avg['search2_time'])],
        ["Fusion Compute",
         "-",
         "-",
         format_time(fusion_avg['fusion_time'])],
        ["TOTAL",
         format_time(oc_avg['total_time']),
         format_time(b3_avg['total_time']),
         format_time(fusion_avg['total_time'])]
    ]
    
    print_table(headers, rows)
    
    return results


# ======================
# Memory Profiling
# ======================

def run_memory_profiling(data_dict):
    """Profile memory usage"""
    print_section_header("GPU MEMORY PROFILING")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping memory profiling.")
        return {}
    
    results = {}
    
    # Baseline
    torch.cuda.empty_cache()
    baseline = get_gpu_memory()
    results['baseline'] = baseline
    print(f"Baseline VRAM (models loaded): {format_memory(baseline)}")
    
    # During image encoding
    sample_image = data_dict['image_paths'][0]
    
    reset_peak_memory()
    _ = encode_image_openclip(data_dict['openclip_model'], data_dict['openclip_preprocess'], sample_image)
    oc_img_mem = get_peak_memory()
    results['openclip_image'] = oc_img_mem
    
    reset_peak_memory()
    _ = encode_image_beit3(data_dict['beit3_model'], data_dict['beit3_preprocess'], sample_image)
    b3_img_mem = get_peak_memory()
    results['beit3_image'] = b3_img_mem
    
    # During text encoding
    sample_query = data_dict['sample_queries'][0]
    
    reset_peak_memory()
    _ = encode_text_openclip(data_dict['openclip_model'], data_dict['openclip_tokenizer'], sample_query)
    oc_text_mem = get_peak_memory()
    results['openclip_text'] = oc_text_mem
    
    reset_peak_memory()
    _ = encode_text_beit3(data_dict['beit3_model'], data_dict['beit3_tokenizer'], sample_query)
    b3_text_mem = get_peak_memory()
    results['beit3_text'] = b3_text_mem
    
    # Print table
    headers = ["Operation", "OpenCLIP", "BEiT3"]
    rows = [
        ["Image Encoding", format_memory(oc_img_mem), format_memory(b3_img_mem)],
        ["Text Encoding", format_memory(oc_text_mem), format_memory(b3_text_mem)]
    ]
    
    print_table(headers, rows, "PEAK VRAM DURING OPERATIONS")
    
    return results


# ======================
# Summary Report
# ======================

def generate_summary_report(all_results):
    """Generate executive summary"""
    print_section_header("EXECUTIVE SUMMARY")
    
    img_results = all_results['image_encoding']
    text_results = all_results['text_encoding']
    search_results = all_results['search']
    
    # Key findings
    print("\n🔍 KEY FINDINGS:\n")
    
    # Image encoding
    oc_throughput = img_results['openclip']['full']['throughput']
    b3_throughput = img_results['beit3']['full']['throughput']
    faster_img = "OpenCLIP" if oc_throughput > b3_throughput else "BEiT3"
    img_speedup = max(oc_throughput, b3_throughput) / min(oc_throughput, b3_throughput)
    print(f"1. Image Encoding: {faster_img} is {img_speedup:.2f}x faster")
    print(f"   - OpenCLIP: {oc_throughput:.2f} images/sec")
    print(f"   - BEiT3: {b3_throughput:.2f} images/sec")
    
    # Text encoding
    oc_text_time = text_results['openclip']['batch']['per_query']
    b3_text_time = text_results['beit3']['batch']['per_query']
    faster_text = "OpenCLIP" if oc_text_time < b3_text_time else "BEiT3"
    text_speedup = max(oc_text_time, b3_text_time) / min(oc_text_time, b3_text_time)
    print(f"\n2. Text Encoding: {faster_text} is {text_speedup:.2f}x faster")
    print(f"   - OpenCLIP: {format_time(oc_text_time)} per query")
    print(f"   - BEiT3: {format_time(b3_text_time)} per query")
    
    # Search latency (k=10)
    k10_results = search_results[10]
    oc_search = np.mean([r['total_time'] for r in k10_results['openclip']])
    b3_search = np.mean([r['total_time'] for r in k10_results['beit3']])
    fusion_search = np.mean([r['total_time'] for r in k10_results['fusion']])
    
    fusion_overhead = ((fusion_search - max(oc_search, b3_search)) / max(oc_search, b3_search)) * 100
    
    print(f"\n3. Search Latency (top-10):")
    print(f"   - OpenCLIP only: {format_time(oc_search)}")
    print(f"   - BEiT3 only: {format_time(b3_search)}")
    print(f"   - Fusion (30-70): {format_time(fusion_search)}")
    print(f"   - Fusion overhead: +{fusion_overhead:.1f}%")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:\n")
    print("1. For real-time applications (<100ms latency):")
    print("   → Use single model (OpenCLIP or BEiT3)")
    print("\n2. For batch processing (accuracy > speed):")
    print("   → Use Fusion for better retrieval quality")
    print("\n3. For resource-constrained environments:")
    if img_results['openclip']['full']['vram'] < img_results['beit3']['full']['vram']:
        print("   → Use OpenCLIP (lower VRAM)")
    else:
        print("   → Use BEiT3 (lower VRAM)")
    
    print("\n" + "="*80)


# ======================
# Main Execution
# ======================

def main():
    print("\n" + "="*80)
    print(" "*25 + "PERFORMANCE BENCHMARK")
    print(" "*20 + "Rank Fusion System Analysis")
    print("="*80)
    
    # System info
    print(f"\nSystem Information:")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  Batch Size: {batch_size}")
    
    # Load everything
    data_dict = load_models_and_data()
    
    all_results = {}
    
    # Run benchmarks
    all_results['image_encoding'] = run_image_encoding_benchmarks(data_dict)
    all_results['text_encoding'] = run_text_encoding_benchmarks(data_dict)
    all_results['search'] = run_search_benchmarks(data_dict)
    all_results['memory'] = run_memory_profiling(data_dict)
    
    # Generate summary
    generate_summary_report(all_results)
    
    print("\n✅ Benchmark complete!\n")


if __name__ == "__main__":
    main()
