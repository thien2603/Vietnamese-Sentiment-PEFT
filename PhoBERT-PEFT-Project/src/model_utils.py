from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

def create_model(model_name, num_labels=3, use_lora=False, lora_rank=8):
    """
    Tải model gốc và (tùy chọn) bọc nó bằng LoRA.
    """
    print(f"\n--- Đang tạo model: {model_name} ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )

    if use_lora:
        print(f"Áp dụng cấu hình LoRA (PEFT) với Rank (r) = {lora_rank}...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_rank,
            lora_alpha=lora_rank * 2, # Alpha thường gấp đôi Rank
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
        
        model = get_peft_model(model, peft_config)
        print("--- SO SÁNH THAM SỐ HUẤN LUYỆN ---")
        model.print_trainable_parameters()
    
    return model