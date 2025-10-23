import argparse
import time
import pandas as pd
from transformers import TrainingArguments, Trainer

# Import các hàm đã module hóa từ thư mục src
from src.data_processing import load_and_preprocess_data, compute_metrics
from src.model_utils import create_model

def run_experiment(args):
    """
    Hàm chính để chạy toàn bộ pipeline huấn luyện và đánh giá.
    """
    
    # --- 1. TẢI DATA VÀ MODEL ---
    tokenized_datasets, tokenizer = load_and_preprocess_data(
        dataset_name=args.dataset_name, 
        model_name=args.model_name
    )
    
    model = create_model(
        model_name=args.model_name, 
        use_lora=args.use_lora,
        lora_rank=args.lora_rank # Truyền rank vào
    )

    # --- 2. CẤU HÌNH HUẤN LUYỆN ---
    print("\n--- Đang cấu hình TrainingArguments ---")
    
    # Chọn learning rate dựa trên phương pháp
    lr = args.learning_rate_lora if args.use_lora else args.learning_rate_full
    print(f"Sử dụng Learning Rate: {lr}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=lr, # Sử dụng LR đã chọn
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
        fp16=True,      
        report_to="none"
    )

    # --- 3. KHỞI TẠO TRAINER ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # --- 4. HUẤN LUYỆN ---
    print("\n" + "="*50)
    if args.use_lora:
        print(f" BẮT ĐẦU HUẤN LUYỆN: LoRA (PEFT) r={args.lora_rank}")
    else:
        print(" BẮT ĐẦU HUẤN LUYỆN: Full Fine-Tuning")
    print("="*50)
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time = end_time - start_time

    # --- 5. ĐÁNH GIÁ VÀ BÁO CÁO ---
    print("\n--- Đang đánh giá trên tập Test ---")
    results = trainer.evaluate(tokenized_datasets["test"])
    
    f1_score = results["eval_f1_macro"]
    accuracy = results["eval_accuracy"]
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "="*50)
    print(f" KẾT QUẢ THÍ NGHIỆM: {args.output_dir}")
    print(f"  Test F1-macro: {f1_score:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Thời gian huấn luyện: {training_time:.2f}s")
    print(f"  Số tham số huấn luyện: {trainable_params:,}")
    print("="*50)

if __name__ == "__main__":
    # --- 6. PARSE THAM SỐ DÒNG LỆNH ---
    parser = argparse.ArgumentParser(description="So sánh Full Fine-tuning vs LoRA (PEFT) cho PhoBERT.")
    
    # Tham số chung
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base")
    parser.add_argument("--dataset_name", type=str, default="uitnlp/vietnamese_students_feedback")
    parser.add_argument("--output_dir", type=str, required=True, help="Thư mục để lưu model (ví dụ: results/baseline)")
    
    # Tham số Huấn luyện
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate_full", type=float, default=2e-5, help="LR cho Full Fine-tuning")
    parser.add_argument("--learning_rate_lora", type=float, default=1e-4, help="LR cho LoRA (thường cao hơn)")
    
    # Tham số LoRA
    parser.add_argument("--use_lora", action="store_true", help="Kích hoạt LoRA (PEFT)")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank (r) cho LoRA")

    args = parser.parse_args()
    
    run_experiment(args)