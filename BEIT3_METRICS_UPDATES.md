# BEiT3 Training Metrics Updates

## Tổng quan thay đổi

Đã thêm các metrics mới vào quá trình finetuning BEiT3 cho retrieval tasks:

### ✅ Metrics mới được thêm vào:

1. **eval_loss**: Loss trong quá trình validation/evaluation
2. **tr_mean_rank**: Mean rank cho Text-to-Image retrieval
3. **tr_median_rank**: Median rank cho Text-to-Image retrieval
4. **ir_mean_rank**: Mean rank cho Image-to-Text retrieval
5. **ir_median_rank**: Median rank cho Image-to-Text retrieval

### 📊 Tensorboard Visualization

Các metrics được tổ chức theo nhóm trong Tensorboard:

- **eval/**: eval_loss, average_score
- **eval/text_to_image/**: r1, r5, r10, mean_rank, median_rank
- **eval/image_to_text/**: r1, r5, r10, mean_rank, median_rank

---

## Files đã sửa đổi

### 1. `beit3/engine_for_finetuning.py`

#### Thay đổi trong `RetrievalHandler`:

- **`__init__()`**: Thêm `self.criterion` để tính contrastive loss trong evaluation
- **`eval_batch()`**: Tính eval_loss và log vào metric_logger
- **`after_eval()`**: 
  - Tính mean_rank và median_rank cho text-to-image retrieval
  - Tính mean_rank và median_rank cho image-to-text retrieval
  - Thêm các metrics mới vào `eval_result` dictionary

### 2. `beit3/run_beit3_finetuning.py`

#### Thay đổi trong main training loop:

- Sau khi evaluate(), log tất cả metrics vào Tensorboard
- Metrics được nhóm theo category (eval, eval/text_to_image, eval/image_to_text)
- Chỉ áp dụng cho retrieval tasks (flickr30k, coco_retrieval)

---

## Cách sử dụng

### Training mới:

```bash
# Chạy training như bình thường, metrics mới sẽ tự động được log
python beit3/run_beit3_finetuning.py \
    --task flickr30k \
    --log_dir ./logs \
    --output_dir ./checkpoints \
    ... (các args khác)
```

### Xem kết quả trong Tensorboard:

```bash
tensorboard --logdir=./logs
```

Truy cập http://localhost:6006 để xem:
- **SCALARS** tab → **eval/** → Xem tất cả evaluation metrics
- **eval/text_to_image/** → Metrics cho text→image retrieval
- **eval/image_to_text/** → Metrics cho image→text retrieval

---

## Ví dụ output trong log.txt

```json
{
  "train_lr": 0.0001,
  "train_loss": 0.5,
  "val_loss": 0.45,
  "val_tr_r1": 30.5,
  "val_tr_r5": 58.3,
  "val_tr_r10": 70.2,
  "val_tr_mean_rank": 5.8,
  "val_tr_median_rank": 3.0,
  "val_ir_r1": 28.8,
  "val_ir_r5": 56.0,
  "val_ir_r10": 68.7,
  "val_ir_mean_rank": 6.2,
  "val_ir_median_rank": 4.0,
  "val_average_score": 52.08,
  "epoch": 10
}
```

---

## Lưu ý

- **Rank values**: Càng thấp càng tốt (ideal = 1.0)
- **R@K values**: % precision, càng cao càng tốt (max = 100.0)
- **Mean rank**: Nhạy cảm với outliers
- **Median rank**: Robust hơn với outliers

---

## Testing checklist

- [x] Code compile không lỗi syntax
- [ ] Training chạy thành công từ đầu đến cuối
- [ ] Log file chứa đầy đủ metrics mới
- [ ] Tensorboard hiển thị đúng graphs
- [ ] Mean/median rank có giá trị hợp lý (>= 1.0)

---

**Ngày cập nhật**: 2025-11-06
**Người thực hiện**: AI Assistant (theo yêu cầu của Bui Tung Hung)
