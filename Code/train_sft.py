# save as: train_sft_qwen25vl.py
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

# -------- 配置区域 --------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_MIN_PIXELS = 256*28*28
IMAGE_MAX_PIXELS = 1280*28*28   # 你也可以用默认动态分辨率
USE_FLASH_ATTENTION_2 = True

USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]  # 语言侧
# 如需调视觉塔/投影，可扩展 target_modules；详见社区 finetune 方案。 

OUTPUT_DIR = "outputs/qwen25vl_sft"
EPOCHS = 1
LR = 1e-4
BATCH_PER_DEVICE = 1
GRAD_ACCUM = 8
MAX_SAMPLES = None  # 开发时可设较小数调通

# -------- 数据加载（LLaVA 格式）--------
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

# -------- collator：把 conversations+images -> 模型输入 --------
def build_collate_fn(processor):
    def collate(batch: List[Dict[str, Any]]):
        messages_list = []
        image_inputs_all, video_inputs_all = [], []
        for ex in batch:
            conv = ex["conversations"]
            # 转成 chat messages（严格遵循模型卡示例）
            # human 含 <image> 占位，但真正的像素由 process_vision_info 传入
            messages = []
            for turn in conv:
                role = "user" if turn["from"] == "human" else "assistant"
                content = []
                # 解析 human value 里 <image> 个数，与 images 对齐
                if role == "user":
                    img_count = turn["value"].count("<image>")
                    # image 实际路径（file://）
                    imgs = ex["images"][:img_count]
                    for p in imgs:
                        content.append({"type":"image","image": f"file://{os.path.join(ex['image_folder'], p)}"})
                    # 去掉占位文本，仅保留文字提示
                    text = turn["value"].replace("<image>","").strip()
                    if text:
                        content.append({"type":"text","text": text})
                else:
                    content.append({"type":"text","text": turn["value"]})
                messages.append({"role": role, "content": content})

            messages_list.append(messages)

        # 文本模板
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_list
        ]
        # 图像/视频张量
        vision_infos = [process_vision_info(m) for m in messages_list]
        images = [vi[0] for vi in vision_infos]  # list of list
        videos = [vi[1] for vi in vision_infos]

        # processor 负责把文本 + 图像打包成张量；padding 到 batch 最大长度
        model_inputs = processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        # labels：仅监督 assistant 输出（SFT）
        # 这里我们简化：让所有 token 参与 LM 训练（通常也可用 completion-only 策略）
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
    return collate

def main(train_json: str, image_folder: str):
    # 模型 & 处理器
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
        remove_unused_columns=False,   # 多模态必须关
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=processor,   # TRL SFTTrainer 会在必要时处理 tokenizer/特殊符号
        data_collator=build_collate_fn(processor),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    # 示例：使用 AITZ 转换后的数据
    main("data/llava/aitz_train.json", "path/to/aitz_images")
