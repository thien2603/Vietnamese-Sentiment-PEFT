# Phân tích So sánh Full Fine-Tuning và LoRA (PEFT) cho PhoBERT

Đây là project nghiên cứu so sánh hiệu suất và hiệu quả của bốn phương pháp fine-tuning mô hình `vinai/phobert-base` trên bài toán Phân loại Cảm xúc Tiếng Việt (bộ dữ liệu 3 nhãn: NEG, NEU, POS).

## 1. Mục tiêu Project

Nghiên cứu này thực hiện một chuỗi thí nghiệm có hệ thống để trả lời các câu hỏi:
* **Hiệu quả (Efficiency):** Kỹ thuật LoRA (PEFT) giúp tiết kiệm tài nguyên (số tham số, thời gian huấn luyện) đến mức nào so với Full Fine-Tuning?
* **Hiệu suất (Performance):** Chúng ta có phải hy sinh độ chính xác (F1-score) để đổi lấy hiệu quả không?
* **Điểm tối ưu (Sweet Spot):** Rank (r) của LoRA bằng bao nhiêu (`r=8, 16, hay 32`) là tối ưu nhất cho bài toán này?

## 2. Kết quả So sánh 📊

Đây là bảng kết quả cuối cùng, được tổng hợp từ các lần chạy thí nghiệm.

| Phương pháp | Test F1-macro | Thời gian train (s) | Số tham số train | Tỉ lệ tham số |
|:---|---:|---:|:---|---:|
| 1. Full Fine-Tuning (Baseline) | 0.7124 | 479.52 | 135,000,579 | 100.00% |
| 2. LoRA (PEFT) r=8 | 0.6571 | **70.50** | 1,035,267 | **0.77%** |
| 3. LoRA (PEFT) r=256 | **0.7127** | 104.58 | 14,748,675 | 10.92% |
| 4. LoRA (PEFT) r=512 | 0.7064 | 133.33 | 28,904,451 | 21.41% |

*(Lưu ý: Bạn nên cập nhật bảng này với kết quả cuối cùng chính xác nhất của mình)*

## 3. Phân tích & Kết luận

* **LoRA r=8:** Nhanh nhất nhưng hiệu suất thấp, bị underfit.
* **LoRA r=16 (Điểm ngọt 🏆):** Đạt F1-score tương đương Baseline, nhanh hơn 4.6 lần, chỉ dùng 10.9% tham số.
* **LoRA r=32:** Hiệu suất giảm nhẹ, tốn tài nguyên hơn r=16.

**Kết luận:** Thí nghiệm chứng minh **LoRA (r=16)** là chiến lược tối ưu, cân bằng hoàn hảo giữa hiệu suất và hiệu quả cho bài toán này.

---
## 4. Cách chạy lại (How to Run) 🚀

Có 2 cách để chạy lại các thí nghiệm trong project này:

### Cách 1: Chạy bằng Script `.py` (Chuyên nghiệp)

Cách này phù hợp nếu bạn muốn chạy thí nghiệm trên máy local hoặc server có GPU.

**Bước 1: Chuẩn bị Môi trường**

1.  Clone repository này về máy của bạn:
    ```bash
    git clone [URL-repo-GitHub-của-bạn]
    cd [Tên-repo-của-bạn]
    ```
2.  (Khuyến nghị) Tạo và kích hoạt môi trường ảo (virtual environment):
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```
3.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

**Bước 2: Chạy Thí nghiệm**

Sử dụng script `train.py` với các tham số dòng lệnh:

* **Chạy Baseline (Full Fine-Tuning):**
    ```bash
    python train.py --output_dir "results/baseline"
    ```
* **Chạy LoRA với Rank=16:**
    ```bash
    python train.py --output_dir "results/lora_r16" --use_lora --lora_rank 16
    ```
* **Chạy LoRA với Rank=32:**
    ```bash
    python train.py --output_dir "results/lora_r32" --use_lora --lora_rank 32
    ```
* **(Tùy chọn)** Chạy với các tham số khác (ví dụ: 5 epochs, batch size 16):
    ```bash
    python train.py --output_dir "results/lora_r16_5epochs" --use_lora --lora_rank 16 --num_epochs 5 --batch_size 16
    ```

**Lưu ý:** Script sẽ tự động tải dataset từ Hugging Face. Đảm bảo bạn có kết nối Internet ổn định.

---
### Cách 2: Chạy bằng Notebook `.ipynb` trên Google Colab (Trực quan) 🧪

Cách này phù hợp để xem lại toàn bộ quá trình thí nghiệm, output chi tiết và chạy nhanh trên GPU miễn phí của Google.

**Bước 1: Mở Google Colab**

1.  Truy cập [colab.research.google.com](https://colab.research.google.com/).
2.  Chọn **`File`** -> **`Upload notebook...`** (Tải sổ tay lên...).
3.  Chọn tab **`GitHub`**.
4.  Dán **URL của repo GitHub** này vào ô tìm kiếm và nhấn Enter.
5.  Chọn file notebook (ví dụ: `PhoBERT_PEFT_Analysis.ipynb`) từ danh sách.

**Bước 2: Chọn Runtime GPU**

1.  Sau khi notebook mở ra, chọn **`Runtime`** (Thời gian chạy) trên thanh menu.
2.  Chọn **`Change runtime type`** (Thay đổi loại thời gian chạy).
3.  Trong mục "Hardware accelerator" (Trình tăng tốc phần cứng), chọn **`T4 GPU`** (hoặc GPU khác nếu có).
4.  Nhấn **`Save`** (Lưu).

**Bước 3: Chạy Notebook**

1.  Chọn **`Runtime`** (Thời gian chạy) trên thanh menu.
2.  Chọn **`Run all`** (Chạy tất cả).

Notebook sẽ tự động cài đặt thư viện, tải data, chạy lần lượt 4 thí nghiệm (Baseline, LoRA r=8, r=16, r=32) và in ra bảng so sánh kết quả cuối cùng. Quá trình này có thể mất khoảng 20-30 phút.
