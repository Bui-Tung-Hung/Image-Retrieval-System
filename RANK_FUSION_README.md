# Rank Fusion for Image-Text Retrieval

Hệ thống kết hợp sức mạnh của OpenCLIP và BEiT3 để cải thiện độ chính xác trong image retrieval.

## 📋 Tổng quan

System này bao gồm 3 scripts chính:

1. **rank_fusion_encode.py** - Encode tất cả images và tạo FAISS indices
2. **rank_fusion_evaluation.py** - Đánh giá metrics trên test set
3. **rank_fusion_demo.py** - Demo interactive cho người dùng

## 🔧 Cài đặt

### Requirements
```bash
pip install torch torchvision
pip install open-clip-torch
pip install transformers
pip install faiss-cpu  # hoặc faiss-gpu
pip install pandas tqdm pillow numpy
```

### Cấu trúc thư mục
```
D:\Projects\doan\
├── open_clip/
│   └── checkpoints/
│       └── epoch_15.pt
├── beit3/
│   ├── beit3.spm
│   ├── modeling_finetune.py
│   └── utils.py
├── rank_fusion_encode.py
├── rank_fusion_evaluation.py
├── rank_fusion_demo.py
└── rank_fusion_output/  (sẽ được tạo tự động)

D:\Projects\kaggle\
├── test/  (folder chứa 2535 images)
└── test_corrected.csv  (ground truth)

C:\Users\LAPTOP\Downloads\BEiT3\
└── ckpt\
    └── checkpoint-best.pth
```

## 🚀 Hướng dẫn sử dụng

### Bước 1: Encode images (chạy 1 lần duy nhất)

```bash
cd D:\Projects\doan
python rank_fusion_encode.py
```

Script này sẽ:
- Load 2 models (OpenCLIP và BEiT3)
- Encode tất cả 2535 images bằng cả 2 models
- Tạo 2 FAISS indices
- Lưu kết quả vào `rank_fusion_output/`

**Thời gian ước tính**: ~5-10 phút (tùy GPU)

**Output files**:
- `openclip_embeddings.pt` - OpenCLIP embeddings (512-dim)
- `beit3_embeddings.pt` - BEiT3 embeddings (768-dim)
- `openclip_image_index.faiss` - FAISS index cho OpenCLIP
- `beit3_image_index.faiss` - FAISS index cho BEiT3
- `image_paths.pkl` - Danh sách đường dẫn images

### Bước 2: Evaluation (đánh giá metrics)

```bash
python rank_fusion_evaluation.py
```

Script này sẽ:
- Load ground truth từ CSV
- Evaluate 3 configurations:
  - OpenCLIP only (100%)
  - BEiT3 only (100%)
  - Fusion (30% OpenCLIP + 70% BEiT3)
- Tính toán metrics: R@1, R@5, R@10, Mean Rank, Median Rank
- In bảng so sánh
- Lưu kết quả vào `evaluation_results.json`

**Thời gian ước tính**: ~10-20 phút (tùy số lượng captions)

**Output**:
```
================================================================================
EVALUATION RESULTS COMPARISON
================================================================================
Model                          R@1       R@5      R@10   Mean Rank  Median Rank
--------------------------------------------------------------------------------
OpenCLIP Only                 XX.XX%    XX.XX%    XX.XX%       XX.XX         XX
BEiT3 Only                    XX.XX%    XX.XX%    XX.XX%       XX.XX         XX
Fusion (30-70)                XX.XX%    XX.XX%    XX.XX%       XX.XX         XX
================================================================================
```

### Bước 3: Interactive Demo

```bash
python rank_fusion_demo.py
```

Script này cho phép:
- Nhập query text tự do
- Xem kết quả fusion (30% + 70%)
- So sánh kết quả giữa 3 models

**Ví dụ sử dụng**:
```
Enter your query: Một con chó đang chạy trên bãi cỏ
🔍 Searching for: 'Một con chó đang chạy trên bãi cỏ'

Fusion Results (30% OpenCLIP + 70% BEiT3)
Rank   Image Name                                         Score
----------------------------------------------------------------------
1      dog_running_001.jpg                                0.8532
2      dog_grass_045.jpg                                  0.8421
3      puppy_field_123.jpg                                0.8198
...

Show comparison with individual models? (y/n): y
```

## 📊 Metrics giải thích

- **R@K (Recall at K)**: Tỷ lệ % queries có ground truth image xuất hiện trong top-K kết quả
- **Mean Rank**: Trung bình vị trí của ground truth image
- **Median Rank**: Vị trí trung vị của ground truth image

**Ví dụ**: 
- R@1 = 45% → 45% queries có đúng ảnh ở vị trí số 1
- Mean Rank = 3.2 → Trung bình ground truth ở vị trí thứ 3.2

## 🔬 Fusion Strategy

**Formula**: `fusion_score = 0.3 × cosine_sim_openclip + 0.7 × cosine_sim_beit3`

**Lý do chọn 30-70**:
- BEiT3 được train trên data lớn hơn → weight cao hơn
- OpenCLIP vẫn giữ 30% để balance và tận dụng điểm mạnh

**Để thử weights khác**, sửa trong code:
```python
# Trong rank_fusion_evaluation.py
results['Custom Fusion'] = evaluate_model(
    models_dict, ground_truth,
    weight1=0.5, weight2=0.5,  # Thay đổi ở đây
    model_name="Fusion (50-50)"
)
```

## ⚙️ Tùy chỉnh

### Thay đổi batch size (nếu bị out of memory)
```python
# Trong rank_fusion_encode.py, dòng 23
batch_size = 16  # Giảm từ 32 xuống 16
```

### Thay đổi số lượng kết quả trả về
```python
# Trong rank_fusion_demo.py
results = search_with_fusion(query, models_dict, top_k=10)  # Thay 5 thành 10
```

## 🐛 Troubleshooting

**Lỗi: "CUDA out of memory"**
- Giảm `batch_size` trong `rank_fusion_encode.py`
- Hoặc chuyển sang CPU: `device = "cpu"`

**Lỗi: "File not found"**
- Kiểm tra lại đường dẫn trong configuration section của từng script
- Đảm bảo đã chạy `rank_fusion_encode.py` trước khi chạy 2 scripts còn lại

**Lỗi: "No module named 'open_clip'"**
- Cài đặt: `pip install open-clip-torch`

**Lỗi: "Can't load BEiT3 checkpoint"**
- Kiểm tra path: `C:\Users\LAPTOP\Downloads\BEiT3\ckpt\checkpoint-best.pth`
- Đảm bảo file tồn tại và có quyền đọc

## 📈 Kết quả kỳ vọng

Dựa trên nghiên cứu, fusion thường cải thiện:
- R@1: +2-5%
- R@5: +3-7%
- Mean Rank: Giảm 10-20%

Tuy nhiên kết quả phụ thuộc vào:
- Chất lượng của từng model
- Data distribution
- Fusion weights

## 📝 Notes

- **Chạy encode 1 lần**: Sau khi đã encode xong, không cần chạy lại trừ khi thay đổi images hoặc models
- **Ground truth format**: CSV với delimiter `;`, format `image_filename;caption`
- **Multiple captions**: Mỗi image có thể có nhiều captions (5 captions per image trong Flickr8k)
- **FAISS IndexFlatIP**: Sử dụng Inner Product cho cosine similarity (embeddings đã normalized)

## 🔍 Phân tích thêm

Để phân tích sâu hơn, bạn có thể:

1. **Visualize confusion cases**: Images được rank sai bởi fusion
2. **Per-category analysis**: Phân tích theo category (người, động vật, phong cảnh...)
3. **Weight tuning**: Thử nhiều weight combinations khác (20-80, 40-60, 50-50...)
4. **Add more models**: Mở rộng fusion với 3+ models

## 📧 Support

Nếu gặp vấn đề, kiểm tra lại:
1. File paths trong configuration
2. Đã cài đặt đủ dependencies
3. Đã chạy encode script trước
4. GPU memory đủ (hoặc chuyển sang CPU)

---
Created with ❤️ for Image Retrieval Research
