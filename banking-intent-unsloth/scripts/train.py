import yaml
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, DataCollatorForSeq2Seq


def main():
    # Đọc file cấu hình YAML
    with open("configs/train.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    max_seq_length = config["data"]["max_seq_length"]

    print("\n=== KHỞI TẠO MÔ HÌNH VÀ TOKENIZER VỚI UNSLOTH ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config["model"]["name_or_path"],
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = config["model"]["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = config["model"]["lora_r"],
        target_modules = config["model"]["target_modules"],
        lora_alpha = config["model"]["lora_alpha"],
        lora_dropout = config["model"]["lora_dropout"],
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = config["training"].get("seed", 3407),
    )

    print("\n=== TẢI DỮ LIỆU ===")
    train_dataset = load_dataset("csv", data_files=config["data"]["train_path"])["train"]
    
    # Định dạng dữ liệu thành mẫu Prompt-Completion (Đầu vào - Đầu ra)
    prompt_template = "Intent classification. Input: {}\nLabel: {}"
    
    def format_dataset(examples):
        texts = []
        for text, label in zip(examples["text"], examples["label"]):
            # Thêm eos_token ở cuối để mô hình biết điểm kết thúc câu
            texts.append(prompt_template.format(text, label) + tokenizer.eos_token)
        return { "text_formatted" : texts }

    train_dataset = train_dataset.map(format_dataset, batched=True)


    print("\n=== HUẤN LUYỆN ===")
    sft_config = SFTConfig(
        output_dir = config["output"]["output_dir"],
        per_device_train_batch_size = config["training"]["batch_size"],
        gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"],
        learning_rate = float(config["training"]["learning_rate"]),
        num_train_epochs = config["training"]["num_train_epochs"],
        optim = config["training"]["optimizer"],
        weight_decay = config["training"]["weight_decay"],
        lr_scheduler_type = config["training"]["lr_scheduler_type"],
        warmup_steps = config["training"]["warmup_steps"],
        logging_steps = config["output"]["logging_steps"],
        seed = config["training"].get("seed", 3407),
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        report_to = "none",

        dataset_text_field = "text_formatted",
        max_length = max_seq_length,
        dataset_num_proc = 4,
        packing = False,
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset = train_dataset,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        args = sft_config,
        processing_class = tokenizer,
    )

    trainer.train()
    
    print(f"\n=== LƯU MÔ HÌNH VÀO {config['output']['output_dir']} ===")
    model.save_pretrained(config["output"]["output_dir"])
    tokenizer.save_pretrained(config["output"]["output_dir"])
    print("Hoàn tất Fine-tuning!")

if __name__ == "__main__":
    main()