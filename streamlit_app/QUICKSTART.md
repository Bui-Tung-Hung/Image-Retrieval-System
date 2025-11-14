# Quick Start Guide

## Installation

1. **Navigate to app directory:**
   ```bash
   cd d:\Projects\doan\streamlit_app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify setup:**
   ```bash
   python test_setup.py
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## First Time Usage

### Step 1: Encode Your Images

1. Open app at http://localhost:8501
2. Go to **📂 Encode Images** tab
3. Enter your image folder path (e.g., `D:/my_images`)
4. Select **Both** models
5. Click **Encode Folder**
6. Wait for encoding to complete

### Step 2: Search

1. Go to **🔍 Search** tab
2. Select **Fusion** in sidebar
3. Set fusion weight to `0.5`
4. Enter a query like "a cat"
5. Click **Search**

## Common Issues

### "Module not found" error
- Make sure you're in the `streamlit_app` directory
- Check that BEiT3 folder is one level up

### "CUDA out of memory"
- Edit `config.py` and reduce `BATCH_SIZE` to 16 or 8
- Or install `faiss-cpu` instead of `faiss-gpu`

### Images not showing
- Verify image paths are still valid
- Don't move images after encoding
- Re-encode if you moved the images

## Tips

- **Encode with Both**: Always encode with both models for maximum flexibility
- **Start with Fusion**: Fusion mode usually gives best results
- **Adjust Weight**: Try different fusion weights (0.0-1.0) to see what works best
- **GPU Recommended**: Encoding and search are much faster with GPU
