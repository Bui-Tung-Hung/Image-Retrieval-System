# Image Retrieval System - Streamlit UI

A powerful image retrieval application using OpenCLIP and BEiT3 models with FAISS indexing and rank fusion capabilities.

## Features

- 🔍 **Text-to-Image Search**: Search for images using natural language queries
- 🤖 **Multiple Models**: Use OpenCLIP, BEiT3, or Fusion of both
- 📂 **Batch Encoding**: Encode entire folders or individual files
- 🗑️ **Image Management**: View and delete images from indices
- 🔄 **Incremental Updates**: Add images without rebuilding entire index
- 💾 **Auto-Save**: All changes are saved immediately
- 🎯 **UUID-Based**: Robust image tracking using UUIDs

## Installation

### 1. Install Dependencies

```bash
cd d:\Projects\doan\streamlit_app
pip install -r requirements.txt
```

**Note**: Use `faiss-gpu` if you have CUDA-capable GPU, otherwise use `faiss-cpu`.

### 2. Verify Model Paths

Check `config.py` and ensure all model paths are correct:

- `OPENCLIP_MODEL_PATH`: Path to your finetuned OpenCLIP checkpoint
- `BEIT3_MODEL_PATH`: Path to BEiT3 base weights
- `BEIT3_TOKENIZER_PATH`: Path to BEiT3 sentencepiece tokenizer
- `BEIT3_CHECKPOINT_PATH`: Path to finetuned BEiT3 checkpoint

### 3. Run Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage Guide

### Tab 1: 🔍 Search

1. **Select Model** in sidebar:
   - **OpenCLIP**: Fast, good for general images
   - **BEiT3**: Better for Vietnamese captions
   - **Fusion**: Best results, combines both models

2. **Adjust Fusion Weight** (if Fusion selected):
   - `0.0` = 100% BEiT3
   - `0.5` = Equal weight (default)
   - `1.0` = 100% OpenCLIP

3. **Enter Query** and set **Top K** results

4. **Click Search** to retrieve images

### Tab 2: 📂 Encode Images

#### Encode Folder

1. Enter folder path (e.g., `D:/images/my_dataset`)
2. Select model: **Both** (recommended), **OpenCLIP**, or **BEiT3**
3. Click **Encode Folder**

The system will:
- Recursively scan for images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`)
- Encode with selected model(s)
- Add to FAISS index
- Save automatically

#### Encode Files

1. Click **Browse files** to upload images
2. Select multiple images
3. Choose model: **Both**, **OpenCLIP**, or **BEiT3**
4. Click **Encode Uploaded Files**

### Tab 3: 🗑️ Manage Images

1. Select index: **OpenCLIP** or **BEiT3**
2. Use **Filter** to search by path
3. Check images to delete (or use **Select All**)
4. Click **Delete Selected**
5. Confirm deletion

**Note**: Deleting from one index doesn't affect the other.

### Tab 4: ⚙️ Settings

- View model and index paths
- Adjust default fusion weight
- Clear all indices (⚠️ dangerous!)

## Architecture

### Directory Structure

```
streamlit_app/
├── app.py                  # Main Streamlit application
├── config.py              # Configuration constants
├── models.py              # Model loading & caching
├── faiss_manager.py       # FAISS index management
├── image_encoder.py       # Image encoding utilities
├── search_engine.py       # Search & rank fusion
├── ui_components.py       # Reusable UI components
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── data/
    ├── indices/          # FAISS indices & metadata
    │   ├── openclip.index
    │   ├── beit3.index
    │   └── metadata.json
    └── uploads/          # Temporary uploaded files
```

### FAISS Index Architecture

Uses **IndexIDMap2** with **IndexFlatIP** base:

- **IndexFlatIP**: Inner Product (cosine similarity after L2 normalization)
- **IndexIDMap2**: Maps UUIDs → vectors, supports incremental add/remove
- **UUID System**: Each image gets unique UUID, stored as int64 in FAISS

### Metadata Structure

`metadata.json` stores mappings:

```json
{
  "openclip": {
    "images": [
      {
        "uuid": "550e8400-e29b-41d4-a716-446655440000",
        "path": "D:\\images\\cat.jpg",
        "added_at": "2025-11-07T10:30:00",
        "faiss_index": 0
      }
    ],
    "uuid_to_index": {...},
    "path_to_uuid": {...},
    "total_images": 1
  },
  "beit3": {...}
}
```

### Rank Fusion Algorithm

**Reciprocal Rank Fusion (RRF)**:

```
score(image) = α × RRF_openclip + (1-α) × RRF_beit3

RRF(rank) = 1 / (k + rank)
```

Where:
- `α` = fusion weight (0.0 to 1.0)
- `k` = 60 (constant from config)
- `rank` = position in result list

## API Reference

### ModelManager

```python
manager = ModelManager()
model, preprocess, tokenizer = manager.get_openclip()
wrapper = manager.get_beit3()
```

### FAISSManager

```python
faiss_mgr = FAISSManager('openclip', 512)

# Add images
faiss_mgr.add_vectors(vectors, image_paths)

# Search
distances, uuids = faiss_mgr.search(query_vector, k=20)

# Remove images
faiss_mgr.remove_vectors(uuids_to_remove)

# Get all images
images = faiss_mgr.get_all_images()
```

### ImageEncoder

```python
encoder = ImageEncoder(model_manager)

# Encode folder
vectors, paths = encoder.encode_folder('/path/to/folder', 'openclip')

# Encode files
vectors, paths = encoder.encode_files(['img1.jpg', 'img2.jpg'], 'beit3')
```

### SearchEngine

```python
search = SearchEngine(model_manager, faiss_managers)

# Search with single model
scores, uuids = search.search_openclip("a cat", k=20)

# Search with fusion
fusion_results = search.search_fusion("a cat", k=20, alpha=0.5)
```

## Troubleshooting

### Model Loading Errors

**Problem**: `FileNotFoundError` when loading models

**Solution**: Check paths in `config.py`, ensure all checkpoint files exist

### CUDA Out of Memory

**Problem**: GPU memory error during encoding

**Solution**: 
- Reduce `BATCH_SIZE` in `config.py` (default: 32)
- Use `faiss-cpu` instead of `faiss-gpu`
- Encode smaller batches

### Images Not Found

**Problem**: Images display as "not found" after moving files

**Solution**: 
- Don't move images after encoding
- Or re-encode from new location
- UUIDs remain valid, paths become invalid

### Index Corruption

**Problem**: FAISS index won't load

**Solution**:
1. Go to Settings tab
2. Click "Clear All Indices"
3. Re-encode your images

### Slow Search

**Problem**: Search takes too long

**Solution**:
- Reduce `top_k` parameter
- Use single model instead of Fusion
- Ensure using GPU version of FAISS

## Performance Tips

1. **Use GPU**: Install `faiss-gpu` for 10-100x faster search
2. **Batch Encoding**: Encode folders instead of individual files
3. **Cache Models**: Models are cached in session_state (don't restart app frequently)
4. **Fusion Weight**: Start with 0.5, adjust based on results
5. **Index Both Models**: Encode with "Both" option for maximum flexibility

## Hardware Requirements

- **Minimum**: 
  - CPU: 4 cores
  - RAM: 8 GB
  - Storage: 10 GB (for models + indices)

- **Recommended**:
  - CPU: 8+ cores
  - RAM: 16 GB
  - GPU: RTX 3050 Ti or better (4+ GB VRAM)
  - Storage: 50 GB SSD

## License

Part of đồ án tốt nghiệp - Bùi Tùng Hưng

## Support

For issues or questions, check:
1. This README
2. Code comments in source files
3. Original project documentation in parent directory
