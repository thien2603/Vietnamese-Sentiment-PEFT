import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score

def load_and_preprocess_data(dataset_name, model_name, max_length=128):
    """
    Tải, tiền xử lý và token hóa dataset.
    Sử dụng URL trực tiếp để tránh lỗi phiên bản 'datasets'.
    """
    print(f"\n--- Đang tải dataset: {dataset_name} ---")
    try:
        # ✅ SỬA LỖI: Tải trực tiếp từ URL của file CSV
        data_files = {
            "train": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/raw/main/train.csv",
            "validation": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/raw/main/validation.csv",
            "test": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/raw/main/test.csv"
        }
        dataset = load_dataset("csv", data_files=data_files)
        print("Tải dataset (từ file CSV qua URL) thành công!")
    except Exception as e:
        print(f"Lỗi khi tải dataset: {e}")
        raise e

    print(f"\n--- Đang tải tokenizer: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tải tokenizer thành công!")

    def preprocess_function(examples):
        # Đổi tên cột 'sentiment' thành 'label'
        examples["label"] = examples["sentiment"]
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=max_length)

    print("\n--- Đang tiền xử lý (tokenizing) dataset ---")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "sentiment", "topic"])
    print("Tiền xử lý hoàn tất!")
    
    return tokenized_datasets, tokenizer

def compute_metrics(eval_pred):
    """
    Hàm tính toán metrics cho Trainer.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1_macro": f1}