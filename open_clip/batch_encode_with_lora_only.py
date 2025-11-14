"""
Batch encode using CLIP model with LoRA adapters loaded separately.
This script is for when you saved ONLY LoRA weights (--save-lora-only flag).
"""

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
batch_size = 64

image_folder = r"D:\Projects\kaggle\test"

# Path to LoRA-only checkpoint (saved with --save-lora-only)
lora_checkpoint_path = r"D:\Projects\doan\logs\AllData_finetune5_from_ckpt4\All-data-finetune5_from_ckpt4\checkpoints\epoch_20_lora.pt"

# ======================
# Tải model
# ======================
print("Loading LoRA checkpoint...")
lora_checkpoint = torch.load(lora_checkpoint_path, map_location='cpu')

# Lấy LoRA config từ checkpoint
lora_config = lora_checkpoint.get('lora_config', {})
model_name = lora_config.get('model', 'ViT-B-32')
pretrained_name = lora_config.get('pretrained', 'laion2b_s34b_b79k')
lora_r = lora_config.get('r', 16)
lora_alpha = lora_config.get('alpha', 16)
lora_dropout = lora_config.get('dropout', 0.1)
target_modules = lora_config.get('target_modules', 'in_proj,out_proj')

if isinstance(target_modules, str):
    target_modules = target_modules.split(',')

print(f"Model: {model_name}")
print(f"Pretrained: {pretrained_name}")
print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
print(f"Target modules: {target_modules}")

# Load base model
print("\nLoading base model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name=model_name,
    pretrained=pretrained_name
)

# Lock image tower
print("Locking image tower...")
model.lock_image_tower(unlocked_groups=0, freeze_bn_stats=False)

# Lock text tower
print("Locking text tower...")
model.lock_text_tower(unlocked_layers=0, freeze_layer_norm=True)

# Apply LoRA adapters
print(f"Applying LoRA structure...")
lora_blocks = model.apply_lora_to_text_encoder(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules
)
print(f"✅ Applied LoRA to {lora_blocks} transformer blocks")

# Load LoRA weights
print("Loading LoRA weights...")
lora_state_dict = lora_checkpoint['lora_state_dict']

# Load chỉ LoRA parameters
model_state_dict = model.state_dict()
loaded_count = 0
for name, value in lora_state_dict.items():
    if name in model_state_dict:
        model_state_dict[name] = value
        loaded_count += 1
    else:
        print(f"⚠️  LoRA parameter not found in model: {name}")

model.load_state_dict(model_state_dict, strict=False)
print(f"✅ Loaded {loaded_count} LoRA parameters from epoch {lora_checkpoint.get('epoch', 'unknown')}")

# Move to device
model = model.to(device)
model.eval()

# Verify LoRA is enabled
print("\nVerifying LoRA status...")
has_lora = any('lora_' in name for name, _ in model.named_parameters())
print(f"LoRA parameters present: {has_lora}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"LoRA parameters: {lora_params:,}")

# ======================
# Chuẩn bị danh sách ảnh
# ======================
image_list = [
    os.path.join(image_folder, img)
    for img in os.listdir(image_folder)
    if img.lower().endswith((".jpg", ".png", ".jpeg"))
]
print(f"\nFound {len(image_list)} images")

# ======================
# Encode theo batch
# ======================
all_features = []

begin_total = time.time()

for i in tqdm.tqdm(range(0, len(image_list), batch_size), desc="Processing batches"):
    batch_paths = image_list[i:i+batch_size]

    # Load & preprocess
    batch_images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
    batch_tensor = torch.stack(batch_images).to(device)

    with torch.no_grad():
        batch_features = model.encode_image(batch_tensor)
        batch_features /= batch_features.norm(dim=-1, keepdim=True)

    all_features.append(batch_features.cpu())

end_total = time.time()
print(f"\n✅ Processed {len(image_list)} images in {end_total - begin_total:.2f}s")
print(f"Average: {len(image_list)/(end_total - begin_total):.1f} images/s")

# Concatenate
all_features = torch.cat(all_features, dim=0)
print(f"All features shape: {all_features.shape}")

# Save if needed
# torch.save(all_features, "image_features_lora.pt")