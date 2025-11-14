# Sơ Đồ Kiến Trúc Chi Tiết: CLIP ViT-B-16 & BEiT3

> Tài liệu này mô tả chi tiết kiến trúc của hai mô hình đang sử dụng trong dự án Image-Text Retrieval

---

## 📋 Mục Lục

1. [CLIP ViT-B-16 Architecture](#1-clip-vit-b-16-architecture)
   - [Vision Transformer](#vision-transformer-vit-b-16)
   - [Text Transformer](#text-transformer)
   - [Contrastive Learning Flow](#contrastive-learning-flow)
2. [BEiT3 Architecture](#2-beit3-architecture)
   - [Multiway Transformer](#multiway-transformer)
   - [Embedding Modules](#embedding-modules)
   - [Task-Specific Heads](#task-specific-heads)
3. [Rank Fusion System](#3-rank-fusion-system)
4. [Detailed Specifications](#4-detailed-specifications)

---

## 1. CLIP ViT-B-16 Architecture

### Vision Transformer (ViT-B-16)

```mermaid
graph TB
    subgraph "Input Processing"
        A[Input Image<br/>224x224x3] --> B[Patch Embedding<br/>Conv2d: 3→768<br/>kernel=16, stride=16]
        B --> C[Patch Tokens<br/>196 patches<br/>768 dims each]
        C --> D[Add CLS Token<br/>Total: 197 tokens]
    end
    
    subgraph "Position Encoding"
        D --> E[Add Positional Embedding<br/>Learnable 197x768]
        E --> F[Patch Dropout<br/>optional]
    end
    
    subgraph "Pre-Normalization"
        F --> G[Layer Norm Pre<br/>LayerNorm 768]
    end
    
    subgraph "Transformer Encoder (12 Layers)"
        G --> H[Layer 1<br/>ResidualAttentionBlock]
        H --> I[Layer 2-11<br/>10 x ResidualAttentionBlock]
        I --> J[Layer 12<br/>ResidualAttentionBlock]
    end
    
    subgraph "Pooling & Projection"
        J --> K{Pool Type}
        K -->|tok| L[Extract CLS Token<br/>768 dims]
        K -->|avg| M[Average Pool<br/>patches 1-196]
        L --> N[Layer Norm Post<br/>LayerNorm 768]
        M --> N
        N --> O[Linear Projection<br/>768 → 512]
    end
    
    O --> P[Vision Features<br/>512 dims<br/>L2 Normalized]
    
    style A fill:#e1f5ff
    style P fill:#c8e6c9
    style H fill:#fff9c4
    style I fill:#fff9c4
    style J fill:#fff9c4
```

### ResidualAttentionBlock Details

```mermaid
graph TB
    subgraph "ResidualAttentionBlock"
        A[Input x<br/>N x 197 x 768] --> B[Layer Norm 1]
        B --> C[Multi-Head Attention<br/>12 heads, head_dim=64]
        
        subgraph "Multi-Head Attention"
            C --> D[Q, K, V Projection<br/>in_proj_weight: 2304x768]
            D --> E[Split into 12 heads<br/>each: N x 197 x 64]
            E --> F[Scaled Dot-Product<br/>scale = 1/√64]
            F --> G[Softmax]
            G --> H[Attend to Values]
            H --> I[Concat Heads]
            I --> J[Output Projection<br/>768 → 768]
        end
        
        J --> K[Dropout]
        K --> L[Layer Scale<br/>optional]
        A --> M[Residual Connection +]
        L --> M
        
        M --> N[Layer Norm 2]
        N --> O[MLP Block]
        
        subgraph "MLP Block"
            O --> P[Linear 1<br/>768 → 3072<br/>mlp_ratio=4.0]
            P --> Q[GELU Activation]
            Q --> R[Linear 2<br/>3072 → 768]
            R --> S[Dropout]
        end
        
        S --> T[Layer Scale<br/>optional]
        M --> U[Residual Connection +]
        T --> U
    end
    
    U --> V[Output<br/>N x 197 x 768]
    
    style A fill:#e1f5ff
    style V fill:#c8e6c9
    style C fill:#fff9c4
    style O fill:#ffccbc
```

### Text Transformer

```mermaid
graph TB
    subgraph "Input Processing"
        A[Input Text<br/>token IDs] --> B[Token Embedding<br/>Lookup Table<br/>vocab_size=49408<br/>embed_dim=512]
        B --> C[Text Embeddings<br/>N x 77 x 512]
    end
    
    subgraph "Position Encoding"
        C --> D[Add Positional Embedding<br/>Learnable 77x512]
    end
    
    subgraph "Transformer Encoder (12 Layers)"
        D --> E[Layer 1<br/>ResidualAttentionBlock]
        E --> F[Layer 2-11<br/>10 x ResidualAttentionBlock]
        F --> G[Layer 12<br/>ResidualAttentionBlock]
    end
    
    subgraph "Causal Masking"
        H[Causal Attention Mask<br/>77x77 upper triangular]
        H -.->|applied to| E
        H -.->|applied to| F
        H -.->|applied to| G
    end
    
    subgraph "Pooling & Projection"
        G --> I[Layer Norm Final<br/>LayerNorm 512]
        I --> J{Pool Type}
        J -->|argmax| K[Extract EOS Token<br/>512 dims]
        J -->|first| L[Extract First Token]
        J -->|last| M[Extract Last Token]
        K --> N[Text Projection<br/>512 → 512<br/>Parameter Matrix]
        L --> N
        M --> N
    end
    
    N --> O[Text Features<br/>512 dims<br/>L2 Normalized]
    
    style A fill:#e1f5ff
    style O fill:#c8e6c9
    style E fill:#fff9c4
    style F fill:#fff9c4
    style G fill:#fff9c4
```

### Text ResidualAttentionBlock Details

```mermaid
graph TB
    subgraph "Text ResidualAttentionBlock"
        A[Input x<br/>N x 77 x 512] --> B[Layer Norm 1]
        B --> C[Multi-Head Attention<br/>8 heads, head_dim=64]
        
        subgraph "Causal Multi-Head Attention"
            C --> D[Q, K, V Projection<br/>in_proj_weight: 1536x512]
            D --> E[Split into 8 heads<br/>each: N x 77 x 64]
            E --> F[Scaled Dot-Product<br/>scale = 1/√64]
            G[Causal Mask<br/>77x77] -.->|add -inf to future| F
            F --> H[Softmax]
            H --> I[Attend to Values]
            I --> J[Concat Heads]
            J --> K[Output Projection<br/>512 → 512]
        end
        
        K --> L[Dropout]
        L --> M[Layer Scale<br/>optional]
        A --> N[Residual Connection +]
        M --> N
        
        N --> O[Layer Norm 2]
        O --> P[MLP Block]
        
        subgraph "MLP Block"
            P --> Q[Linear 1<br/>512 → 2048<br/>mlp_ratio=4.0]
            Q --> R[GELU Activation]
            R --> S[Linear 2<br/>2048 → 512]
            S --> T[Dropout]
        end
        
        T --> U[Layer Scale<br/>optional]
        N --> V[Residual Connection +]
        U --> V
    end
    
    V --> W[Output<br/>N x 77 x 512]
    
    style A fill:#e1f5ff
    style W fill:#c8e6c9
    style C fill:#fff9c4
    style P fill:#ffccbc
```

### Contrastive Learning Flow

```mermaid
graph TB
    subgraph "Input Batch"
        A[Image Batch<br/>N images] 
        B[Text Batch<br/>N captions]
    end
    
    A --> C[Vision Encoder<br/>ViT-B-16]
    B --> D[Text Encoder<br/>Transformer]
    
    C --> E[Image Features<br/>N x 512<br/>L2 Normalized]
    D --> F[Text Features<br/>N x 512<br/>L2 Normalized]
    
    E --> G[Compute Similarity Matrix<br/>I @ T^T]
    F --> G
    
    G --> H[Logit Scale<br/>exp ln1/0.07 = 14.29]
    H --> I[Scaled Logits<br/>N x N matrix]
    
    subgraph "Loss Computation"
        I --> J[Image-to-Text Loss<br/>Cross-Entropy on rows]
        I --> K[Text-to-Image Loss<br/>Cross-Entropy on cols]
        J --> L[Average Both Losses]
        K --> L
    end
    
    L --> M[CLIP Loss<br/>Symmetric]
    
    subgraph "Training Target"
        N[Ground Truth<br/>Identity Matrix<br/>Diagonal = 1]
        N -.->|target| J
        N -.->|target| K
    end
    
    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style E fill:#c8e6c9
    style F fill:#c8e6c9
    style M fill:#ffcdd2
```

---

## 2. BEiT3 Architecture

### Multiway Transformer

```mermaid
graph TB
    subgraph "Input Modalities"
        A[Vision Input<br/>Image 224x224x3]
        B[Language Input<br/>Text Token IDs]
        C[Vision-Language Input<br/>Image + Text]
    end
    
    subgraph "Embedding Layer"
        A --> D[Vision Embedding<br/>Patch + Position]
        B --> E[Language Embedding<br/>Token + Position]
        C --> F[Unified Embedding<br/>Concat Vision + Language]
    end
    
    subgraph "Multiway Transformer Encoder (12 Layers)"
        D --> G[Encoder Layer 1]
        E --> G
        F --> G
        
        G --> H[Encoder Layer 2-11<br/>10 layers]
        H --> I[Encoder Layer 12]
    end
    
    subgraph "Multiway Layer Structure"
        J[Shared Self-Attention<br/>All modalities]
        K[Vision-specific FFN<br/>Only vision tokens]
        L[Language-specific FFN<br/>Only language tokens]
        J --> K
        J --> L
    end
    
    I --> M{Task Type}
    
    M -->|Retrieval| N[Retrieval Head<br/>Vision + Language CLS]
    M -->|Captioning| O[Captioning Head<br/>MLM Head]
    M -->|VQA| P[VQA Head<br/>Pooler + Classifier]
    M -->|Classification| Q[Classification Head<br/>Vision CLS]
    
    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#e1f5ff
    style N fill:#c8e6c9
    style O fill:#c8e6c9
    style P fill:#c8e6c9
    style Q fill:#c8e6c9
```

### BEiT3 Embedding Modules

```mermaid
graph TB
    subgraph "Vision Embedding"
        A[Input Image<br/>224x224x3] --> B[Patch Embedding<br/>Conv2d: 3→768<br/>kernel=16, stride=16]
        B --> C[Flatten Patches<br/>196 patches x 768]
        C --> D[Add Vision CLS Token<br/>197 tokens]
        D --> E[Add Vision Position Embedding<br/>Sinusoidal or Learnable]
    end
    
    subgraph "Language Embedding"
        F[Input Text<br/>Token IDs] --> G[Token Embedding<br/>SentencePiece Vocab<br/>vocab_size=64010<br/>embed_dim=768]
        G --> H[Add Language CLS Token<br/>at position 0]
        H --> I[Add Language Position Embedding<br/>Learned embeddings]
    end
    
    subgraph "Vision-Language Joint Embedding"
        E --> J[Vision Tokens<br/>197 x 768]
        I --> K[Language Tokens<br/>L x 768]
        J --> L[Concatenate<br/>Vision + Language]
        K --> L
        L --> M[Joint Embedding<br/>197 + L x 768]
    end
    
    E --> N[Vision-Only Path<br/>For Image Tasks]
    I --> O[Language-Only Path<br/>For Text Tasks]
    M --> P[Multimodal Path<br/>For VL Tasks]
    
    style A fill:#e1f5ff
    style F fill:#e1f5ff
    style N fill:#c8e6c9
    style O fill:#c8e6c9
    style P fill:#c8e6c9
```

### BEiT3 Encoder Layer (Multiway)

```mermaid
graph TB
    subgraph "BEiT3 Encoder Layer"
        A[Input Tokens<br/>N x L x 768] --> B[Layer Norm 1]
        
        B --> C[Shared Multi-Head Attention<br/>12 heads, head_dim=64]
        
        subgraph "Shared Self-Attention"
            C --> D[Q, K, V Projection<br/>768 → 768 each]
            D --> E[Split 12 heads<br/>N x L x 64 each]
            E --> F[Scaled Dot-Product<br/>scale = 1/√64]
            F --> G[Softmax]
            G --> H[Attend + Concat]
            H --> I[Output Projection<br/>768 → 768]
        end
        
        I --> J[Dropout + Residual]
        A --> J
        
        J --> K{Multiway Split}
        
        K -->|Vision Tokens| L[Layer Norm 2 Vision]
        K -->|Language Tokens| M[Layer Norm 2 Language]
        
        L --> N[Vision FFN<br/>768 → 3072 → 768<br/>mlp_ratio=4.0]
        M --> O[Language FFN<br/>768 → 3072 → 768<br/>mlp_ratio=4.0]
        
        N --> P[Dropout + Residual]
        O --> Q[Dropout + Residual]
        
        K -->|Vision| P
        K -->|Language| Q
        
        P --> R[Merge Outputs]
        Q --> R
    end
    
    R --> S[Output Tokens<br/>N x L x 768]
    
    style A fill:#e1f5ff
    style S fill:#c8e6c9
    style C fill:#fff9c4
    style N fill:#ffccbc
    style O fill:#ffccbc
```

### Task-Specific Heads

```mermaid
graph TB
    subgraph "BEiT3ForRetrieval"
        A[Encoder Output<br/>N x L x 768] --> B{Split Modalities}
        B -->|Vision CLS| C[Vision Head<br/>Linear 768→768]
        B -->|Language CLS| D[Language Head<br/>Linear 768→768]
        C --> E[L2 Normalize<br/>Vision Features]
        D --> F[L2 Normalize<br/>Language Features]
        E --> G[Cosine Similarity<br/>V @ L^T]
        F --> G
        G --> H[Logit Scale<br/>exp ln1/0.07]
        H --> I[CLIP Loss<br/>Contrastive]
    end
    
    subgraph "BEiT3ForCaptioning"
        J[Encoder Output<br/>with Causal Mask] --> K[Extract Text Tokens<br/>after image_len]
        K --> L[Masked LM Head<br/>Linear 768→64010]
        L --> M[Vocabulary Logits]
        M --> N[Cross-Entropy Loss<br/>vs Ground Truth]
    end
    
    subgraph "BEiT3ForVQA"
        O[Encoder Output] --> P[Pooler<br/>CLS → Dense → Tanh]
        P --> Q[Classifier Head<br/>768 → 1536 → 3129]
        Q --> R[VQA Answer Logits]
    end
    
    subgraph "BEiT3ForImageClassification"
        S[Encoder Output<br/>Vision Only] --> T[Average Pool<br/>patches 1-196]
        T --> U[Layer Norm]
        U --> V[Classification Head<br/>768 → num_classes]
    end
    
    style I fill:#c8e6c9
    style N fill:#c8e6c9
    style R fill=#c8e6c9
    style V fill:#c8e6c9
```

---

## 3. Rank Fusion System

### Fusion Architecture

```mermaid
graph TB
    subgraph "Input"
        A[Query Text<br/>Vietnamese]
        B[Image Database<br/>2535 images]
    end
    
    subgraph "OpenCLIP ViT-B-16 Branch"
        A --> C[OpenCLIP Tokenizer<br/>BPE]
        B --> D[OpenCLIP Vision Encoder<br/>ViT-B-16]
        C --> E[OpenCLIP Text Encoder<br/>Transformer]
        D --> F[Image Embeddings<br/>2535 x 512]
        E --> G[Query Embedding<br/>1 x 512]
        F --> H[FAISS Index<br/>IndexFlatIP]
        G --> I[Search OpenCLIP Index]
        H --> I
        I --> J[OpenCLIP Similarities<br/>2535 scores]
    end
    
    subgraph "BEiT3 Branch"
        A --> K[BEiT3 Tokenizer<br/>SentencePiece]
        B --> L[BEiT3 Vision Encoder<br/>Multiway Transformer]
        K --> M[BEiT3 Text Encoder<br/>Multiway Transformer]
        L --> N[Image Embeddings<br/>2535 x 768]
        M --> O[Query Embedding<br/>1 x 768]
        N --> P[FAISS Index<br/>IndexFlatIP]
        O --> Q[Search BEiT3 Index]
        P --> Q
        Q --> R[BEiT3 Similarities<br/>2535 scores]
    end
    
    subgraph "Rank Fusion"
        J --> S[Normalize Scores<br/>OpenCLIP]
        R --> T[Normalize Scores<br/>BEiT3]
        S --> U[Weight: 0.3<br/>30% OpenCLIP]
        T --> V[Weight: 0.7<br/>70% BEiT3]
        U --> W[Fusion Score<br/>= 0.3*S + 0.7*T]
        V --> W
        W --> X[Sort by Score<br/>Descending]
    end
    
    X --> Y[Top-K Results<br/>Ranked Images]
    
    style A fill:#e1f5ff
    style Y fill:#c8e6c9
    style W fill:#fff9c4
```

### Fusion Evaluation Pipeline

```mermaid
graph TB
    subgraph "Test Dataset"
        A[Flickr8k Vietnamese<br/>1000 images<br/>5 captions each]
    end
    
    subgraph "Ground Truth"
        A --> B[Load test_corrected.csv<br/>Image-Caption Pairs]
    end
    
    subgraph "Evaluation Modes"
        B --> C{Select Mode}
        C -->|Mode 1| D[OpenCLIP Only<br/>100% weight]
        C -->|Mode 2| E[BEiT3 Only<br/>100% weight]
        C -->|Mode 3| F[Fusion<br/>30% + 70%]
    end
    
    subgraph "Metrics Computation"
        D --> G[Search & Rank]
        E --> G
        F --> G
        G --> H[Compute R@1<br/>Top-1 Accuracy]
        G --> I[Compute R@5<br/>Top-5 Accuracy]
        G --> J[Compute R@10<br/>Top-10 Accuracy]
        G --> K[Compute Mean Rank<br/>Average Position]
        G --> L[Compute Median Rank<br/>Median Position]
    end
    
    subgraph "Results"
        H --> M[Comparison Table]
        I --> M
        J --> M
        K --> M
        L --> M
        M --> N[Best Configuration<br/>Highest R@K]
    end
    
    style A fill:#e1f5ff
    style N fill:#c8e6c9
    style F fill:#fff9c4
```

---

## 4. Detailed Specifications

### CLIP ViT-B-16 Specifications

```mermaid
graph LR
    subgraph "Model Specs"
        A[ViT-B-16] --> B[Vision Tower]
        A --> C[Text Tower]
        A --> D[Shared Space]
    end
    
    B --> E[Params: ~86M<br/>Layers: 12<br/>Width: 768<br/>Heads: 12<br/>MLP: 3072<br/>Patches: 16x16]
    
    C --> F[Params: ~63M<br/>Layers: 12<br/>Width: 512<br/>Heads: 8<br/>MLP: 2048<br/>Context: 77]
    
    D --> G[Embedding: 512<br/>Logit Scale: learnable<br/>Loss: InfoNCE]
    
    style A fill:#e3f2fd
    style E fill:#fff9c4
    style F fill:#ffccbc
    style G fill:#c8e6c9
```

### BEiT3 Base Specifications

```mermaid
graph LR
    subgraph "Model Specs"
        A[BEiT3-Base] --> B[Encoder]
        A --> C[Embeddings]
        A --> D[Task Heads]
    end
    
    B --> E[Params: ~222M<br/>Layers: 12<br/>Width: 768<br/>Heads: 12<br/>FFN: 3072<br/>Multiway: Yes]
    
    C --> F[Vision: 768<br/>Language: 768<br/>Vocab: 64010<br/>Patch: 16x16<br/>Max Seq: variable]
    
    D --> G[Retrieval: 768→768<br/>Captioning: MLM<br/>VQA: Pooler+FC<br/>Classification: FC]
    
    style A fill:#e3f2fd
    style E fill:#fff9c4
    style F fill:#ffccbc
    style G fill:#c8e6c9
```

### Parameter Count Breakdown

```mermaid
pie title CLIP ViT-B-16 Parameters (~149M total)
    "Vision Encoder" : 86
    "Text Encoder" : 63
```

```mermaid
pie title BEiT3 Base Parameters (~222M total)
    "Shared Encoder" : 180
    "Vision Embedding" : 10
    "Language Embedding" : 20
    "Task Heads" : 12
```

### Attention Mechanism Comparison

| Component | CLIP ViT-B-16 | BEiT3 Base |
|-----------|---------------|------------|
| **Vision Heads** | 12 heads × 64 dims | 12 heads × 64 dims |
| **Text Heads** | 8 heads × 64 dims | 12 heads × 64 dims |
| **Attention Type** | Bidirectional (Vision)<br/>Causal (Text) | Bidirectional (both)<br/>Causal (captioning only) |
| **QKV Projection** | Separate per modality | Shared in encoder |
| **FFN** | Separate per modality | Multiway (modality-specific) |
| **Position Encoding** | Learnable | Learnable + Sinusoidal |

### Data Flow Dimensions

#### CLIP Vision Path:
```
Input Image (3×224×224)
    ↓ Patch Embedding
Patches (196×768)
    ↓ Add CLS + Pos
Tokens (197×768)
    ↓ Transformer (12 layers)
Features (197×768)
    ↓ Pool CLS
Pooled (768)
    ↓ Project
Output (512) → L2 Normalized
```

#### CLIP Text Path:
```
Input Text (77 token IDs)
    ↓ Token Embedding
Embeddings (77×512)
    ↓ Add Pos
Tokens (77×512)
    ↓ Transformer (12 layers)
Features (77×512)
    ↓ Pool EOS
Pooled (512)
    ↓ Project
Output (512) → L2 Normalized
```

#### BEiT3 Vision Path:
```
Input Image (3×224×224)
    ↓ Patch Embedding
Patches (196×768)
    ↓ Add CLS + Pos
Tokens (197×768)
    ↓ Multiway Transformer (12 layers)
Features (197×768)
    ↓ Extract CLS
Pooled (768)
    ↓ Vision Head
Output (768) → L2 Normalized
```

#### BEiT3 Language Path:
```
Input Text (L token IDs)
    ↓ Token Embedding (SentencePiece)
Embeddings (L×768)
    ↓ Add CLS + Pos
Tokens (L×768)
    ↓ Multiway Transformer (12 layers)
Features (L×768)
    ↓ Extract CLS
Pooled (768)
    ↓ Language Head
Output (768) → L2 Normalized
```

#### BEiT3 Vision-Language Path:
```
Image (3×224×224) + Text (L token IDs)
    ↓ Separate Embeddings
Vision (197×768) + Language (L×768)
    ↓ Concatenate
Joint (197+L×768)
    ↓ Multiway Transformer (12 layers)
Joint Features (197+L×768)
    ↓ Task-specific pooling
Output (task-dependent)
```

---

## 📊 Performance Characteristics

### Computational Complexity

| Model | FLOPs (forward) | Parameters | Memory (fp32) | Inference Speed |
|-------|-----------------|------------|---------------|-----------------|
| **CLIP ViT-B-16** | ~17.6 GFLOPs | 149M | ~600 MB | Fast |
| **BEiT3 Base** | ~22.4 GFLOPs | 222M | ~900 MB | Medium |
| **Rank Fusion** | ~40 GFLOPs | 371M | ~1.5 GB | Medium |

### Training Characteristics

| Aspect | CLIP | BEiT3 |
|--------|------|-------|
| **Pretraining Data** | 400M image-text pairs | Large-scale multimodal |
| **Training Objective** | Contrastive (InfoNCE) | Masked Language/Image Modeling |
| **Batch Size** | Large (32k+) | Medium (2k-8k) |
| **Learning Rate** | 5e-4 | 1e-3 (with warmup) |
| **Optimizer** | AdamW | AdamW |
| **Augmentation** | RandomResizedCrop, ColorJitter | Similar + MIM |

---

## 🎯 Use Cases per Model

### CLIP ViT-B-16
✅ **Best for:**
- Zero-shot image classification
- Fast image-text retrieval
- Cross-modal similarity search
- Real-time applications

⚠️ **Limitations:**
- Lower dimensional embeddings (512)
- Less multilingual capability
- No generative tasks

### BEiT3
✅ **Best for:**
- Multimodal understanding
- Image captioning
- Visual Question Answering
- Fine-grained vision-language tasks

⚠️ **Limitations:**
- Slower inference
- Higher memory usage
- Requires more compute

### Rank Fusion
✅ **Best for:**
- Maximum retrieval accuracy
- Leveraging complementary strengths
- Production systems with quality priority

⚠️ **Limitations:**
- 2x inference cost
- Complex deployment
- Requires both models

---

## 📚 References

- **CLIP Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **BEiT3 Paper**: [Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks](https://arxiv.org/abs/2208.10442)
- **OpenCLIP**: [open_clip GitHub](https://github.com/mlfoundations/open_clip)
- **BEiT3 Code**: [unilm/beit3](https://github.com/microsoft/unilm/tree/master/beit3)

---

**Created**: October 31, 2025  
**Author**: Bui Tung Hung  
**Project**: Vietnamese Image-Text Retrieval with Rank Fusion
