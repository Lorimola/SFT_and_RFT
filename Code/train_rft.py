import argparse
import torch
from datasets import load_from_disk
from transformers import AutoProcessor, TrainingArguments, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig
from trl import DPOTrainer


def filter_has_rejected(ex):
    r = ex.get("rejected_text")
    return isinstance(r, str) and len(r.strip()) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dataset_path", type=str, required=True, help="Path from script.py (save_to_disk)")
    ap.add_argument("--output_dir", type=str, default="./outputs-rft")
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--use_qlora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype) if args.use_qlora else None

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )

    ds = load_from_disk(args.dataset_path)

    needed = {"prompt_text", "chosen_text", "rejected_text"}
    missing = [c for c in needed if c not in ds.column_names]
    if missing:
        raise ValueError(f"Dataset missing required columns for DPO: {missing}. "
                         f"Please run script.py with --emit_text to populate them.")

    ds = ds.filter(filter_has_rejected)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=200,
        report_to="none",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=args.beta,
        train_dataset=ds,
        tokenizer=processor,
        peft_config=peft_config,
        prompt_column="prompt_text",
        chosen_column="chosen_text",
        rejected_column="rejected_text",
    )

    trainer.train()
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    print("DPO complete. Saved to", args.output_dir)


if __name__ == "__main__":
    main()
