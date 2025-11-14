import faiss
import pickle
import torch
import open_clip
from PIL import Image

# ======================
# Load model
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name='ViT-B-16',
    pretrained=r"C:\Users\LAPTOP\Downloads\epoch_15.pt"
)
model = model.to(device).eval()

# ======================
# Load FAISS index + paths
# ======================
index = faiss.read_index("image_index.faiss")
with open("image_paths.pkl", "rb") as f:
    image_list = pickle.load(f)

print("✅ Loaded index with", index.ntotal, "embeddings")

# ======================
# Encode text query
# ======================
def encode_text(text: str):
    with torch.no_grad():
        tokens = open_clip.tokenize([text]).to(device)
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()

query = input("Enter your text query: ")
while query != "exit":
    query_vec = encode_text(query)

    # ======================
    # Search top-k
    # ======================
    k = 5
    scores, indices = index.search(query_vec, k)
    print(f"\nTop-{k} results for '{query}':")
    for rank, idx in enumerate(indices[0]):
        print(f"#{rank+1}: {image_list[idx]} (score={scores[0][rank]:.4f})")
    query = input("\nEnter your text query (or 'exit' to quit): ")