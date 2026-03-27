| `Thiếu thư mục images/masks` | DATASET_ROOT sai hoặc chưa chạy Cell 1 | Kiểm tra path, chạy Cell 1 trước |
| `Không tìm thấy cặp image-mask` | Tên file không khớp (stem khác nhau) | Chạy Cell 1 để chuẩn hóa |
| CUDA out of memory | Batch size quá lớn | Trong script: `per_device_train_batch_size=2` |
| Dataset ít ảnh | < 500 cặp | Giảm epochs xuống 10–15 |

---

## 9. Gợi ý dataset cụ thể

### ISIC 2017
- Slug: `mosharofhossain/isic-2017-skin-lesion-segmentation-dataset`
- Cấu trúc: Thường có `ISIC_xxx.jpg` và `ISIC_xxx_segmentation.png`
- **Cần Cell 1** để map sang images/masks

### Skin Lesion (devdope)
- Slug: `devdope/skin-lesion-dataset-using-segmentation`
- Cấu trúc: `Dataset/Train`, `Val`, `Test` — mỗi split có thư mục theo bệnh (Melanoma, Sarna, ...)
- **Chưa hỗ trợ tự động** — cần kiểm tra cấu trúc bên trong từng thư mục và chuẩn hóa thủ công nếu khác format

---

*Tham chiếu: `kaggle_train_segformer_standalone.py`, `scripts/train_segformer_acne.py`, `KAGGLE_TRAINING_NOTE.md`*
