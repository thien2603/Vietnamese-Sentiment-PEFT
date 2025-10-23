# Vietnamese-Sentiment-PEFT
# Nghiên cứu So sánh Full Fine-Tuning và LoRA (PEFT) cho PhoBERT

Đây là project thực hiện so sánh hiệu suất và hiệu quả của hai phương pháp fine-tuning mô hình `vinai/phobert-base` trên bài toán Phân loại Cảm xúc Tiếng Việt (Dataset: `uitnlp/vietnamese_students_feedback`).

## 1. Mục tiêu Project

Nghiên cứu này tìm cách trả lời các câu hỏi:
* Kỹ thuật LoRA (PEFT) tiết kiệm được bao nhiêu % tài nguyên (tham số huấn luyện) so với Full Fine-Tuning?
* Việc tiết kiệm tài nguyên đó có phải đánh đổi bằng hiệu suất (F1-score) hay không?

## 2. Kết quả Thí nghiệm 📊

Kết quả được thực hiện trên Google Colab (GPU T4) với 3 Epochs, `batch_size=32`.

| Phương pháp | Test F1-macro | Thời gian train (giây) | Số tham số train | Tỉ lệ tham số |
|:---|---:|---:|:---|---:|
| 1. Full Fine-Tuning (Baseline) | **0.8208** | **305.62** | 135,000,579 | 100.00000% |
| 2. LoRA (PEFT) (`r=8`) | 0.6126 | 402.71* | **1,035,267** | **0.76686%** |

*(Bảng này được copy 100% từ output BƯỚC 5 của bạn)*

---

## 3. Phân tích & Nhận xét

Từ bảng kết quả trên, chúng ta có thể rút ra 3 nhận xét quan trọng:

### ✅ 1. Hiệu quả Tham số (Parameter Efficiency)
LoRA đã chứng minh hiệu quả vượt trội trong việc tiết kiệm tài nguyên, chỉ cần huấn luyện **1,035,267** tham số. Con số này chỉ bằng **0.77%** so với 135 triệu tham số của phương pháp Full Fine-Tuning.

### ⚠️ 2. Hiệu suất Mô hình (Model Performance)
Với cài đặt mặc định (`r=8`), hiệu suất của LoRA **giảm sút đáng kể**. Mô hình LoRA chỉ đạt **0.6126 F1-macro**, tương đương **74.63%** hiệu suất so với baseline (0.8208). Điều này cho thấy LoRA không phải lúc nào cũng là giải pháp thay thế trực tiếp mà không cần tinh chỉnh.

### (*) 3. Ghi chú về Thời gian Huấn luyện
Thời gian huấn luyện của LoRA (402.71s) trong thí nghiệm này cao hơn Baseline (305.62s). Đây là một kết quả **bất thường**, gây ra bởi việc script huấn luyện LoRA đã kích hoạt và **chờ người dùng nhập API key của Weights & Biases (W&B)**.

Do đó, phép so sánh về tốc độ trong lần chạy này là **không hợp lệ**.

## 4. Kết luận & Hướng phát triển
* **Kết luận:** LoRA là một kỹ thuật tuyệt vời để tiết kiệm tài nguyên, nhưng việc áp dụng thành công đòi hỏi phải tinh chỉnh siêu tham số (hyperparameter) cẩn thận.
* **Hướng phát triển (Future Work):** Kết quả F1-score thấp của LoRA (0.61) mở ra các hướng thí nghiệm tiếp theo để cải thiện:
    1.  Tăng rank của LoRA (ví dụ: `r=16` hoặc `r=32`).
    2.  Tinh chỉnh `learning_rate` riêng biệt cho LoRA.
    3.  Tăng số `epochs` huấn luyện (vì LoRA hội tụ chậm hơn).

## 5. Cách chạy lại (Reproducibility)
1.  Mở file `[Tên file .ipynb của bạn]` trong Google Colab.
2.  Chọn `Runtime` -> `Change runtime type` -> `T4 GPU`.
3.  Chạy tất cả các cell. (Lưu ý: Để so sánh thời gian chính xác, hãy thêm `report_to="none"` vào `TrainingArguments` của LoRA để tắt W&B).
