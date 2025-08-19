import os, json
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
)
from qwen_vl_utils import process_vision_info
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_MIN_PIXELS = 256*28*28
IMAGE_MAX_PIXELS = 1280*28*28
USE_FLASH_ATTENTION_2 = True

USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

OUTPUT_DIR = "outputs/qwen25vl_sft"
EPOCHS = 1
LR = 1e-4
BATCH_PER_DEVICE = 1
GRAD_ACCUM = 8
MAX_SAMPLES = None


def llava_dataset_from_json(json_path: str, image_folder: str):
    """
    返回 dataset，其中样本为：
      {
        "conversations": [{"from":"human","value":"<image>..."} , {"from":"gpt","value":"..."}],
        "image": "xxx.png" 或 ["a.png","b.png"]
      }
    """
    def gen():
        data = json.load(open(json_path, "r", encoding="utf-8"))
        for ex in data:
            yield {
                "conversations": ex["conversations"],
                "images": ex["image"] if isinstance(ex["image"], list) else [ex["image"]],
                "image_folder": image_folder,
            }
    return load_dataset("json", data_files={"train": json_path})["train"].map(lambda _:_, batched=False)


def build_collate_fn(processor):
    def collate(batch: List[Dict[str, Any]]):
        messages_list = []
        image_inputs_all, video_inputs_all = [], []
        for ex in batch:
            conv = ex["conversations"]
            messages = []
            for turn in conv:
                role = "user" if turn["from"] == "human" else "assistant"
                content = []
                if role == "user":
                    img_count = turn["value"].count("<image>")
                    imgs = ex["images"][:img_count]
                    for p in imgs:
                        content.append({"type":"image","image": f"file://{os.path.join(ex['image_folder'], p)}"})
                    text = turn["value"].replace("<image>","").strip()
                    if text:
                        content.append({"type":"text","text": text})
                else:
                    content.append({"type":"text","text": turn["value"]})
                messages.append({"role": role, "content": content})

            messages_list.append(messages)

        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_list
        ]
        vision_infos = [process_vision_info(m) for m in messages_list]
        images = [vi[0] for vi in vision_infos]
        videos = [vi[1] for vi in vision_infos]

        model_inputs = processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
    return collate

def main(train_json: str, image_folder: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION_2 else "eager",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=IMAGE_MIN_PIXELS,
        max_pixels=IMAGE_MAX_PIXELS,
    )

    if USE_LORA:
        peft_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES
        )
        model = get_peft_model(model, peft_cfg)

    dataset = llava_dataset_from_json(train_json, image_folder)
    if MAX_SAMPLES:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_PER_DEVICE,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=build_collate_fn(processor),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main("data/llava/aitz_train.json", "path/to/aitz_images")
