import argparse
from typing import List, Dict, Any

import torch
from datasets import load_from_disk

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info


def build_collate_fn(processor):
    def collate(batch: List[Dict[str, Any]]):
        texts = [
            processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
            for ex in batch
        ]

        images = []
        for ex in batch:
            imgs = []
            for msg in ex["messages"]:
                if msg["role"] == "user":
                    for c in msg["content"]:
                        if c["type"] == "image":
                            imgs.append(c["image"].replace("file://", ""))
            images.append(imgs)

        model_inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dataset_path", type=str, required=True, help="Path from script.py (save_to_disk)")
    ap.add_argument("--output_dir", type=str, default="./outputs-sft")
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_steps", type=int, default=-1)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    if args.use_lora:
        peft_cfg = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            bias="none", 
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_cfg)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    # load a dataset which is saved by save_to_disk
    ds = load_from_disk(args.dataset_path)

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        save_steps=200,
        bf16=args.bf16,
        remove_unused_columns=False,
        max_steps=args.max_steps,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        processing_class=processor,
        data_collator=build_collate_fn(processor),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("SFT complete. Saved to", args.output_dir)


if __name__ == "__main__":
    main()
