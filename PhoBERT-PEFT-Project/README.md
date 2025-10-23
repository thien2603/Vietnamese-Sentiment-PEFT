# Phân tích So sánh Full Fine-Tuning và LoRA (PEFT) cho PhoBERT

Đây là project nghiên cứu so sánh hiệu suất và hiệu quả của bốn phương pháp fine-tuning mô hình `vinai/phobert-base` trên bài toán Phân loại Cảm xúc Tiếng Việt (bộ dữ liệu 3 nhãn: NEG, NEU, POS).

## 1. Mục tiêu Project

Nghiên cứu này thực hiện một chuỗi thí nghiệm có hệ thống để trả lời các câu hỏi:
* **Hiệu quả (Efficiency):** Kỹ thuật LoRA (PEFT) giúp tiết kiệm tài nguyên (số tham số, thời gian huấn luyện) đến mức nào so với Full Fine-Tuning?
* **Hiệu suất (Performance):** Chúng ta có phải hy sinh độ chính xác (F1-score) để đổi lấy hiệu quả không?
* **Điểm tối ưu (Sweet Spot):** Rank (r) của LoRA bằng bao nhiêu (`r=8, 16, hay 32`) là tối ưu nhất cho bài toán này?

## 2. Kết quả So sánh 📊

Đây là bảng kết quả cuối cùng, được tổng hợp từ 4 lần chạy thí nghiệm.

| Phương pháp | Test F1-macro | Thời gian train (s) | Số tham số train | Tỉ lệ tham số |
|:---|---:|---:|:---|---:|
| 1. Full Fine-Tuning (Baseline) | 0.7124 | 479.52 | 135,000,579 | 100.00% |
| 2. LoRA (PEFT) r=8 | 0.6571 | **70.50** | 1,035,267 | **0.77%** |
| 3. LoRA (PEFT) r=16 | **0.7127** | 104.58 | 14,748,675 | 10.92% |
| 4. LoRA (PEFT) r=32 | 0.7064 | 133.33 | 28,904,451 | 21.41% |

*(Lưu ý: Đây là kết quả từ output của bạn. LoRA r=16 là tốt nhất)*

## 3. Phân tích & Kết luận

* **LoRA r=8:** Nhanh nhất (nhanh hơn 6.8 lần) và nhẹ nhất (0.77% params) nhưng hiệu suất **thấp** (F1=0.657), bị underfit.
* **LoRA r=16 (Điểm ngọt 🏆):** Đạt F1-score **tương đương** (thậm chí nhỉnh hơn) Baseline, trong khi huấn luyện **nhanh hơn 4.6 lần** và chỉ dùng **10.9%** số tham số.
* **LoRA r=32:** Hiệu suất giảm nhẹ (có thể do overfitting), chậm hơn và tốn nhiều tham số hơn `r=16`.

**Kết luận:** Thí nghiệm chứng minh **LoRA (r=16)** là chiến lược tối ưu, cân bằng hoàn hảo giữa hiệu suất và hiệu quả cho bài toán này.

## 4. Cách chạy lại
1. Cài đặt thư viện: `pip install -r requirements.txt`
2. Chạy baseline: `python train.py --output_dir "results/baseline"`
3. Chạy LoRA r=16: `python train.py --output_dir "results/lora_r16" --use_lora --lora_rank 16`