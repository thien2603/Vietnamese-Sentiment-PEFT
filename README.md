# Phân tích So sánh Full Fine-Tuning và LoRA (PEFT) với Rank cao

Đây là project nghiên cứu so sánh hiệu suất và hiệu quả của bốn phương pháp fine-tuning mô hình `vinai/phobert-base` trên bài toán Phân loại Cảm xúc Tiếng Việt.

## 1. Mục tiêu Project

Nghiên cứu này thực hiện một chuỗi thí nghiệm có hệ thống để trả lời các câu hỏi:
* **Hiệu quả (Efficiency):** Kỹ thuật LoRA (PEFT) giúp tiết kiệm tài nguyên (số tham số, thời gian huấn luyện) đến mức nào so với Full Fine-Tuning?
* **Hiệu suất (Performance):** Chúng ta có phải hy sinh độ chính xác (F1-score) để đổi lấy hiệu quả không?
* **Giới hạn (Limit Testing):** Việc tăng Rank (r) của LoRA lên các giá trị rất cao (như **r=256** và **r=512**) có mang lại lợi ích về hiệu suất, hay sẽ dẫn đến lãng phí tài nguyên và overfitting?

## 2. Thí nghiệm

* **Mô hình:** `vinai/phobert-base`
* **Dataset:** Bộ dữ liệu Phân loại Cảm xúc Tiếng Việt (3 nhãn: NEG, NEU, POS).
* **Các thí nghiệm:** 4 lần chạy được thực hiện với cùng số `epochs=3`:
    1.  **Run 1: Full Fine-Tuning (Baseline)** - Huấn luyện toàn bộ ~135 triệu tham số.
    2.  **Run 2: LoRA (PEFT) với Rank = 8**
    3.  **Run 3: LoRA (PEFT) với Rank = 256**
    4.  **Run 4: LoRA (PEFT) với Rank = 512**

## 3. Kết quả So sánh 📊

Đây là bảng kết quả cuối cùng, được tổng hợp từ 4 lần chạy thí nghiệm.

[!!! DÁN BẢNG KẾT QUẢ (OUTPUT CỦA BƯỚC 5) VÀO ĐÂY !!!]

*Ví dụ template bảng:*
| Phương pháp | Test F1-macro | Thời gian train (s) | Số tham số train | Tỉ lệ tham số |
|:---|---:|---:|:---|---:|
| Full Fine-Tuning (Baseline) | [Điền F1] | [Điền Time] | [Điền Params] | 100.00% |
| LoRA r=8 | [Điền F1] | [Điền Time] | [Điền Params] | [Điền %] |
| LoRA r=256 | [Điền F1] | [Điền Time] | [Điền Params] | [Điền %] |
| LoRA r=512 | [Điền F1] | [Điền Time] | [Điền Params] | [Điền %] |

---

## 4. Phân tích & Kết luận

*(Sau khi có kết quả, bạn hãy phân tích. Rất có thể bạn sẽ rơi vào một trong hai kịch bản sau):*

**[KỊCH BẢN A: Nếu F1 của r=256/512 TĂNG CAO HƠN r=8]**
* **Kết luận:** Thí nghiệm cho thấy `r=8` là quá nhỏ (underfit). Việc tăng rank lên `r=256` đã giúp mô hình học được nhiều thông tin hơn, dẫn đến F1-score tăng [Ghi F1] và đạt [XX]% hiệu suất của baseline, trong khi chỉ huấn luyện [YY]% số tham số.

**[KỊCH BẢN B: Nếu F1 của r=256/512 BẰNG HOẶC GIẢM so với r=8 (hoặc một rank nhỏ hơn)]**
* **Kết luận:** Thí nghiệm cho thấy `r=8` (hoặc một rank nhỏ nào đó) đã là "điểm ngọt" (sweet spot). Việc tăng rank lên `r=256` và `r=512` không mang lại lợi ích về F1-score mà còn làm tăng đáng kể số tham số huấn luyện (lên đến [YY]% so với baseline) và tăng thời gian train. Đây là bằng chứng của việc lãng phí tài nguyên (diminishing returns) hoặc overfitting.

## 5. Cách chạy lại (Reproducibility)
1.  Tạo thư mục project, đặt các file `train.csv`, `validation.csv` (hoặc `val.csv`), `test.csv` vào.
2.  Đặt file Notebook (hoặc file `train.py`) vào cùng thư mục.
3.  Mở Notebook trong Google Colab (chọn `Runtime` -> `T4 GPU`) hoặc chạy `train.py` từ terminal.
4.  (Nếu dùng Notebook) Chạy tất cả các cell.
