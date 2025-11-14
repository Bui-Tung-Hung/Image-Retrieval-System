import torch
from PIL import Image
import os
import sys
import time
from tqdm import tqdm
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model

# Add beit3 folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modeling_finetune
import utils

# ======================
# Cấu hình
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64  # bạn có thể chỉnh tùy theo VRAM của GPU
image_folder = r"D:\kaggle\archive\Info_Retrieval_DS_Split\open_vilc\test"
checkpoint_path = r"D:\Projects\doan\beit3\beit3_base_itc_patch16_224.pth"
sentencepiece_path = r"D:\Projects\doan\beit3\beit3.spm"


def load_beit3_model():
    """Load BEiT3 model for retrieval with pretrained checkpoint"""
    print("Loading BEiT3 model...")
    
    # Create model
    model = create_model(
        'beit3_base_patch16_224_retrieval',
        pretrained=False,
        drop_path_rate=0.1,
        vocab_size=64010,
        checkpoint_activations=None,
    )
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        utils.load_model_and_may_interpolate(
            ckpt_path=checkpoint_path,
            model=model,
            model_key='model|module',
            model_prefix=''
        )
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    return model


# ======================
# Tải model
# ======================
model = load_beit3_model()

# ======================
# Tạo transforms
# ======================
preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),  # bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
])

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

for i in tqdm(range(0, len(image_list), batch_size), desc="Processing batches"):
    batch_paths = image_list[i:i+batch_size]

    # Load & preprocess ảnh trong batch
    batch_images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
    batch_tensor = torch.stack(batch_images).to(device)

    with torch.no_grad():
        begin = time.time()
        # BEiT3 forward with only_infer=True returns (vision_cls, language_cls)
        # vision_cls is already normalized
        vision_cls, _ = model(image=batch_tensor, only_infer=True)
        batch_features = vision_cls
        print(f"Batch {i//batch_size + 1} time:", time.time() - begin)

    all_features.append(batch_features.cpu())

end_total = time.time()
print(f"processed {len(image_list)} in ✅ Total time:", end_total - begin_total, "seconds")

# Nối toàn bộ embedding thành 1 tensor
all_features = torch.cat(all_features, dim=0)
print("All features shape:", all_features.shape)  # (num_images, embedding_dim)

# # Lưu ra file nếu cần
# torch.save(all_features, "image_features_beit3.pt")
# print("Saved embeddings to image_features_beit3.pt")
