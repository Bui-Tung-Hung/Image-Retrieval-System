"""
Script chuẩn bị dataset Flickr8k Vietnamese cho fine-tuning CLIP
"""

from datasets import load_dataset
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

# Tải dataset
print("Đang tải dataset từ Hugging Face...")
dataset = load_dataset("Veinnn/flickr8k-vietnamese04")

# Tạo thư mục lưu trữ
os.makedirs("data/flickr8k_vi/images", exist_ok=True)

# Chuẩn bị dữ liệu
data_rows = []

print("Đang chuẩn bị dữ liệu...")
for idx, example in enumerate(tqdm(dataset['train'])):
    # Lưu ảnh
    image = example['image']
    image_filename = example['image_filename']
    image_path = f"data/flickr8k_vi/images/{image_filename}"
    
    # Lưu ảnh nếu chưa tồn tại
    if not os.path.exists(image_path):
        image.save(image_path)
    
    # Tạo cặp image-caption (tiếng Việt)
    captions_vi = example['captions_vi']
    
    for caption in captions_vi:
        data_rows.append({
            'filepath': image_path,
            'title': caption
        })

# Tạo CSV file
df = pd.DataFrame(data_rows)
df.to_csv('data/flickr8k_vi/metadata.csv', index=False)

print(f"✅ Đã tạo dataset với {len(df)} cặp image-caption")
print(f"📁 CSV file: data/flickr8k_vi/metadata.csv")
print(f"🖼️ Thư mục ảnh: data/flickr8k_vi/images/")
print(f"\nĐầu vài dòng dữ liệu:")
print(df.head())