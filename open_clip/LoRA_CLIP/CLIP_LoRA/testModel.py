import open_clip
import torch
from PIL import Image

# Tải model và preprocessing
# model, _, preprocess = open_clip.create_model_and_transforms(
#     'ViT-B-32',
#     pretrained='laion2b_s34b_b79k'
# )

# Hoặc tải từ local checkpoint
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name='ViT-B-32',
    pretrained=r'D:\Projects\doan\logs\flickr8k_vi5\flickr8k_vi_finetune\checkpoints\epoch_2.pt'
)

model.eval()
import os 
image_folder = r"D:\kaggle\archive\Info_Retrieval_DS_Split\open_vilc\test"

image_list = []
for image_name in os.listdir(image_folder):
    if image_name.endswith(".jpg"):
        image_list.append(os.path.join(image_folder, image_name))


import time
print(f"Found Total {len(image_list)} images")
begin1 = time.time()
import tqdm
for image in tqdm.tqdm(image_list, desc="Processing images"):
    image = preprocess(Image.open(image)).unsqueeze(0)
    with torch.no_grad():
        begin = time.time()
        image_features = model.encode_image(image)
        print("time", time.time()-begin)


        
        # text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # print(f"Similarity: {similarity}")

end1 = time.time()
print("Total time", end1-begin1)