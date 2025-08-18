from trl import DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
import torch

def train_dpo(model_name="/data1/models/Qwen2.5-VL-7B-Instruct", dataset_path="your_rft_dataset.json", output_dir="/data2/home/donglingzhong/yangsb/SAR/models/RFT"):

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    # Load dataset: needs 'prompt', 'chosen', 'rejected' fields
    dataset = load_dataset("json", data_files=dataset_path)

    # Optional: tokenize all in advance
    def preprocess(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    processed_dataset = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)

    # TrainingArguments
    args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_train_epochs=3,
        fp16=True,
        logging_steps=10,
        output_dir=output_dir,
        save_strategy="epoch",
        report_to="none",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 默认用原始模型作为参考
        args=args,
        beta=0.1,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
    )

    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)

if __name__ == "__main__":
    train_dpo()
