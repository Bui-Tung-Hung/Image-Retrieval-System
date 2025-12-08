# Image Retrieval System - Streamlit UI

An image retrieval application using OpenCLIP and BEiT3 models with FAISS indexing and rank fusion capabilities.

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
