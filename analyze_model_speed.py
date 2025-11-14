"""
Phân tích tại sao SigLIP2 chậm hơn CLIP
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'open_clip', 'src'))

import torch
import open_clip
import time
from PIL import Image
import numpy as np

print("="*70)
print("PHÂN TÍCH TỐC ĐỘ ENCODING: CLIP vs SigLIP2")
print("="*70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# ======================
# CLIP ViT-B-16
# ======================
print("\n" + "="*70)
print("1. CLIP ViT-B-16 (OpenCLIP)")
print("="*70)

model1, _, preprocess1 = open_clip.create_model_and_transforms(
    'ViT-B-16',
    pretrained=False
)
model1 = model1.to(device).eval()

# Count parameters
vision_params1 = sum(p.numel() for p in model1.visual.parameters())
total_params1 = sum(p.numel() for p in model1.parameters())

print(f"Vision tower parameters: {vision_params1:,}")
print(f"Total parameters: {total_params1:,}")
print(f"Image size: 224x224")
print(f"Embedding dimension: 512")

# Test encoding speed
dummy_img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img_tensor1 = preprocess1(dummy_img1).unsqueeze(0).to(device)

# Warmup
for _ in range(5):
    with torch.no_grad():
        _ = model1.encode_image(img_tensor1)

# Benchmark
times1 = []
for _ in range(100):
    start = time.perf_counter()
    with torch.no_grad():
        _ = model1.encode_image(img_tensor1)
    if device == "cuda":
        torch.cuda.synchronize()
    times1.append(time.perf_counter() - start)

avg_time1 = np.mean(times1) * 1000  # Convert to ms
print(f"Avg encoding time: {avg_time1:.2f} ms/image")
print(f"Throughput: {1000/avg_time1:.1f} images/sec")
print(f"For 2535 images: {2535 * avg_time1 / 1000:.1f} seconds = {2535 * avg_time1 / 60000:.1f} minutes")

# ======================
# SigLIP2 SO400M
# ======================
print("\n" + "="*70)
print("2. SigLIP2 ViT-SO400M-16-384")
print("="*70)

print("Loading SigLIP2 (this may take a while)...")
model2, _, preprocess2 = open_clip.create_model_and_transforms(
    'ViT-SO400M-16-SigLIP2-384',
    pretrained='webli'  # Need to load pretrained to test real performance
)
model2 = model2.to(device).eval()
print("✓ Loaded")

# Count parameters
vision_params2 = sum(p.numel() for p in model2.visual.parameters())
total_params2 = sum(p.numel() for p in model2.parameters())

print(f"Vision tower parameters: {vision_params2:,}")
print(f"Total parameters: {total_params2:,}")
print(f"Image size: 384x384")
print(f"Embedding dimension: 1152")

# Test encoding speed
dummy_img2 = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
img_tensor2 = preprocess2(dummy_img2).unsqueeze(0).to(device)

# Warmup
print("Warming up...")
for _ in range(5):
    with torch.no_grad():
        _ = model2.encode_image(img_tensor2)

# Benchmark
print("Benchmarking...")
times2 = []
for _ in range(100):
    start = time.perf_counter()
    with torch.no_grad():
        _ = model2.encode_image(img_tensor2)
    if device == "cuda":
        torch.cuda.synchronize()
    times2.append(time.perf_counter() - start)

avg_time2 = np.mean(times2) * 1000  # Convert to ms
print(f"Avg encoding time: {avg_time2:.2f} ms/image")
print(f"Throughput: {1000/avg_time2:.1f} images/sec")
print(f"For 2535 images: {2535 * avg_time2 / 1000:.1f} seconds = {2535 * avg_time2 / 60000:.1f} minutes")

# ======================
# COMPARISON
# ======================
print("\n" + "="*70)
print("PHÂN TÍCH SO SÁNH")
print("="*70)

print(f"\n1. Số lượng parameters:")
print(f"   CLIP vision:    {vision_params1:,}")
print(f"   SigLIP2 vision: {vision_params2:,}")
print(f"   → SigLIP2 lớn hơn {vision_params2/vision_params1:.1f}x")

print(f"\n2. Kích thước ảnh:")
print(f"   CLIP:    224 x 224 = {224*224:,} pixels")
print(f"   SigLIP2: 384 x 384 = {384*384:,} pixels")
print(f"   → SigLIP2 nhiều hơn {(384*384)/(224*224):.1f}x pixels")

print(f"\n3. Tốc độ encoding:")
print(f"   CLIP:    {avg_time1:.2f} ms/image ({1000/avg_time1:.1f} imgs/sec)")
print(f"   SigLIP2: {avg_time2:.2f} ms/image ({1000/avg_time2:.1f} imgs/sec)")
print(f"   → SigLIP2 chậm hơn {avg_time2/avg_time1:.1f}x")

print(f"\n4. Thời gian encode 2535 images:")
print(f"   CLIP:    {2535 * avg_time1 / 60000:.1f} phút")
print(f"   SigLIP2: {2535 * avg_time2 / 60000:.1f} phút")
print(f"   → Chênh lệch: {(2535 * avg_time2 - 2535 * avg_time1) / 60000:.1f} phút")

print("\n" + "="*70)
print("KẾT LUẬN")
print("="*70)

speedup_from_params = vision_params2 / vision_params1
speedup_from_pixels = (384*384) / (224*224)
actual_slowdown = avg_time2 / avg_time1

print(f"\nDự đoán slowdown:")
print(f"  - Từ parameters: {speedup_from_params:.1f}x")
print(f"  - Từ image size: {speedup_from_pixels:.1f}x")
print(f"  - Tổng hợp (ước tính): ~{speedup_from_params * speedup_from_pixels:.1f}x")
print(f"\nThực tế slowdown: {actual_slowdown:.1f}x")

if actual_slowdown > 20:
    print("\n⚠ CẢNH BÁO: Tốc độ quá chậm!")
    print("Nguyên nhân có thể:")
    print("  1. Model quá lớn (SO400M = 400 triệu parameters)")
    print("  2. Image resolution cao (384x384)")
    print("  3. Không tối ưu hóa inference")
    print("  4. GPU memory bandwidth bottleneck")
    print("\nGiải pháp:")
    print("  1. Giảm batch_size xuống 8 hoặc 4")
    print("  2. Sử dụng torch.compile() để tối ưu")
    print("  3. Sử dụng FP16/BF16 thay vì FP32")
    print("  4. Cân nhắc sử dụng model nhỏ hơn (ViT-L-16-SigLIP2)")
elif actual_slowdown > 10:
    print("\n✓ Slowdown trong mức chấp nhận được")
    print("  Model lớn hơn nhiều nên chậm hơn là bình thường")
else:
    print("\n✓ Performance tốt!")

print("="*70)
