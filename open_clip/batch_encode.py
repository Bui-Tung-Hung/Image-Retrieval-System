import open_clip
import torch
from PIL import Image
import os
import tqdm
import time

# ======================
# Cấu hình
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64  # bạn có thể chỉnh tùy theo VRAM của GPU
image_folder = r"D:\Projects\kaggle\test"

# ======================
# Tải model và preprocess
# ======================
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name='ViT-B-16',
    pretrained=r"C:\Users\LAPTOP\Downloads\epoch_15.pt"
)
model = model.to(device)
model.eval()

# ======================
# Chuẩn bị danh sách ảnh
# ======================
image_list = [
    os.path.join(image_folder, img)
    for img in os.listdir(image_folder)
    if img.lower().endswith((".jpg", ".png", ".jpeg"))
]
print(f"Found Total {len(image_list)} images")

# ======================
# Encode theo batch
# ======================
all_features = []

begin_total = time.time()

for i in tqdm.tqdm(range(0, len(image_list), batch_size), desc="Processing batches"):
    batch_paths = image_list[i:i+batch_size]

    # Load & preprocess ảnh trong batch
    batch_images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
    batch_tensor = torch.stack(batch_images).to(device)

    with torch.no_grad():
        batch_features = model.encode_image(batch_tensor)
        batch_features /= batch_features.norm(dim=-1, keepdim=True)
        #print(f"Batch {i//batch_size + 1} time:", time.time() - begin)

    all_features.append(batch_features.cpu())

end_total = time.time()
print(f"processed {len(image_list)} in ✅ Total time:", end_total - begin_total, "seconds")

# Nối toàn bộ embedding thành 1 tensor
all_features = torch.cat(all_features, dim=0)
print("All features shape:", all_features.shape)  # (num_images, embedding_dim)

# # Lưu ra file nếu cần
# torch.save(all_features, "image_features.pt")
# print("Saved embeddings to image_features.pt")
import faiss
import numpy as np
import pickle
embedding_dim = all_features.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(all_features.numpy())

# Lưu FAISS index ra file
faiss.write_index(index, "faiss/image_index.faiss")
print("✅ Saved FAISS index to image_index.faiss")

# Lưu danh sách file ảnh để mapping lại khi search
with open("faiss/image_paths.pkl", "wb") as f:
    pickle.dump(image_list, f)
print("✅ Saved image paths to image_paths.pkl")