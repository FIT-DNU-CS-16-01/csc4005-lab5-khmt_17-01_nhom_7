# CSC4005 Lab 5 Report – Vision Transformer for Smart Campus Scene Classification

## 1. Thông tin nhóm/cá nhân

- Họ tên: Nguyễn Văn Tiến
- Mã sinh viên: 1771040025
- Lớp: KHMT_17-01
- Link GitHub repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab5-khmt_17-01_nhom_7
- Link W&B dashboard: https://wandb.ai/tienxinhtrai20k-dainam-vietnam/csc4005-lab5-mit-indoor-vit?nw=nwusertienxinhtrai20k

## 2. Mô tả bài toán

Bài toán: phân loại ảnh nội thất trong môi trường trường học vào 5 lớp: `classroom`, `computerroom`, `library`, `corridor`, `office`.

Ứng dụng: hệ thống Smart Campus có thể tự động nhận biết loại không gian để hỗ trợ giám sát, quản lý tài nguyên và phân tích hành vi. ViT phù hợp vì nó học được các đặc trưng toàn cảnh (layout, đồ đạc, cấu trúc không gian) thông qua patch embeddings và cơ chế attention.

## 3. Dữ liệu

| Nội dung | Mô tả |
|---|---|
| Dataset gốc | MIT Indoor Scenes 67 |
| Subset sử dụng | classroom, computerroom, library, corridor, office |
| Số ảnh mỗi lớp | classroom: 113, computerroom: 114, library: 107, corridor: 346, office: 109 |
| Train/Val/Test split | Total: 789 → train: 557, val: 116, test: 116 (stratified, seed=42) |
| Tiền xử lý | Resize 224×224, normalize (ImageNet mean/std). Augmentations (when `--augment`): horiz. flip, small rotation, color jitter. |

## 4. Mô hình ViT

Mô tả ngắn gọn: image → patch embedding → positional embedding → transformer encoder → classification head

| Thành phần | Giá trị |
|---|---|
| model_name | `vit_b_16` |
| train_mode | `head_only` (frozen backbone) và `finetune` (toàn bộ) |
| img_size | 224 |
| batch size | head_only: 16, finetune: 8 |
| số epoch | head_only: 10, finetune: 5 |
| learning rate | head_only: 0.001, finetune: 5e-05 |
| optimizer | AdamW |
| total params | 85,802,501 |
| trainable params | head_only: 3,845; finetune: 85,802,501 |
| trainable ratio | head_only: 4.48e-05; finetune: 1.0 |

## 5. Kết quả

| Run | Best val (metric) | Test accuracy | Test macro-F1 | Best epoch |
|---|---:|---:|---:|---:|
| `vit_b16_head_only` | val macro-F1 = 0.93211 | 0.91379 | 0.88107 | 10 |
| `vit_b16_finetune` | val macro-F1 = 0.88949 | 0.93966 | 0.91305 | 5 |

Artifacts (in-repo):
- `outputs/vit_b16_head_only/` — `config.json`, `metrics.json`, `history.csv`, `curves.png`, `confusion_matrix.png`, `best_model.pt`, `class_to_idx.json`
- `outputs/vit_b16_finetune/` — same files for the finetune run

(Chèn ảnh learning curves và confusion matrices từ các file `curves.png` và `confusion_matrix.png` trong thư mục outputs tương ứng.)

## 6. Phân tích lỗi

1. Lớp dự đoán tốt nhất/ kém nhất:
- Dựa trên macro-F1 và accuracy, `finetune` cải thiện tổng thể so với `head_only`. Không có số liệu per-class trực tiếp trong file metrics.json, xem `confusion_matrix.png` để biết chi tiết per-class.

2. Cặp lớp dễ nhầm:
- Các lớp có bối cảnh tương tự (ví dụ `classroom` ↔ `office`, `library` ↔ `computerroom`) thường có nhầm lẫn nhiều hơn. Kiểm tra `outputs/*/confusion_matrix.png` để trích mọi nhầm lẫn cụ thể.

3. Mất cân bằng dữ liệu:
- `corridor` có số lượng lớn (346 ảnh) so với các lớp khác (~100 ảnh), có thể gây bias; cân nhắc undersample/oversample hoặc trọng số lớp.

4. Ảnh hưởng của augmentation:
- Các run đã bật `--augment`. Augmentation giúp giảm overfitting và ổn định learning curves; vẫn có thể thử MixUp/CutMix/random crop để tăng tính khái quát.

5. Đề xuất cải thiện:
- Cân bằng dataset (undersample/oversample) hoặc dùng weighted loss.
- Thử augmentation mạnh hơn và regularization.
- Dùng mixed precision (AMP) để tăng batch size khả dụng và tốc độ huấn luyện.
- Fine-tune lâu hơn hoặc sử dụng LR scheduling nếu tài nguyên cho phép.

## 7. Liên hệ với lý thuyết ViT

1. Patch embedding tương tự token embedding trong NLP: ảnh được chia patch, flatten và ánh xạ sang vector embedding.
2. Positional embedding cung cấp thông tin vị trí cho mô hình transformer, vì self-attention bản thân nó không xử lý thông tin vị trí.
3. `head_only` nhanh hơn vì chỉ cập nhật một số lượng nhỏ tham số (chỉ head), nên ít tốn thời gian và VRAM; `finetune` cập nhật toàn bộ mô hình.
4. Nên fine-tune toàn bộ khi dataset có kích thước đủ lớn hoặc khác biệt rõ so với dữ liệu pretraining, và khi có GPU/ thời gian huấn luyện.

## 8. W&B evidence

- Project: `csc4005-lab5-mit-indoor-vit` (runs were logged when `--use_wandb` was used).
- Lưu ý: mỗi run lưu đầy đủ hyperparameters (xem `config.json`), các metric theo epoch (xem `history.csv`) và artefact ảnh trong `outputs/<run_name>`.

## 9. Kết luận

- Kết quả: ViT (`vit_b_16`) đạt test accuracy ≈ 91.4% (head-only) và ≈ 93.97% (finetune). Finetune cho macro-F1 cao hơn (≈0.913) so với head-only (≈0.881).
- Nhận xét: với dataset nhỏ, `head_only` là lựa chọn tiết kiệm tài nguyên nhưng `finetune` mang lại hiệu suất tốt hơn nếu có GPU mạnh.
- Hành động tiếp theo: cân bằng dữ liệu, áp dụng augmentation/phép tiền xử lý mạnh hơn, sử dụng AMP và LR scheduling cho finetune để tìm thêm cải thiện.

---

_Artifacts và số liệu chi tiết nằm trong thư mục `outputs/vit_b16_head_only` và `outputs/vit_b16_finetune`._
