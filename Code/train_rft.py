"""
DPO 对 Qwen2.5-VL-7B-Instruct 的偏好微调 (RFT)
- 依赖: transformers>=4.51.3, trl>=0.12.*, peft, datasets, accelerate, pillow
- 数据要求: 每条样本包含
    images: List[PIL.Image.Image] 或 图像路径(本脚本会自动打开)
    prompt: 已对话模板化前的 user 内容(包含一个或多张 <image>)
    chosen: 助手更优回答
    rejected: 助手较差回答
- 若是图像路径，请把字段命名为 image_paths: List[str] 或 image_path: str
"""

import io
import os
import argparse
from typing import List, Dict, Any

import torch
from datasets import load_dataset, Dataset, Features, Sequence, Value, Image as HFImage
from PIL import Image

from transformers import (
    AutoProcessor,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import DPOTrainer

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenForVL
    _QWEN_CLASS = QwenForVL
except Exception:
    from transformers import AutoModelForCausalLM as _QWEN_CLASS


def ensure_images_in_memory(example: Dict[str, Any], image_root: str = "") -> Dict[str, Any]:
    """
    将 image_path(s) 转为 PIL，并标准化为 `images: List[PIL.Image]`。
    兼容:
      - images: 已是 PIL 列表
      - image_path: 单路径
      - image_paths: 路径列表
    """
    if "images" in example and isinstance(example["images"], list):
        # 可能是已经 decode 的 PIL，也可能是 {bytes/...}
        imgs = []
        for im in example["images"]:
            if isinstance(im, Image.Image):
                imgs.append(im)
            elif isinstance(im, dict) and "bytes" in im:
                imgs.append(Image.open(io.BytesIO(im["bytes"])))  # 兜底
        example["images"] = imgs
        return example

    paths = None
    if "image_paths" in example:
        paths = example["image_paths"]
    elif "image_path" in example:
        paths = [example["image_path"]]

    if paths is not None:
        imgs = []
        for p in paths:
            p = p if (image_root == "" or os.path.isabs(p)) else os.path.join(image_root, p)
            imgs.append(Image.open(p).convert("RGB"))
        example["images"] = imgs
    return example


def format_for_chat(example: Dict[str, Any], processor: AutoProcessor, max_resize: int = None) -> Dict[str, Any]:
    """
    将 (images, prompt, chosen, rejected) 按 VLM 聊天模板转换为文本串，让 TRL 的 DPOTrainer 能用。
    """
    # 可选：限制长边，避免显存暴涨
    if max_resize is None:
        try:
            max_resize = processor.image_processor.size.get("longest_edge", None)
        except Exception:
            max_resize = None

    imgs: List[Image.Image] = example["images"]
    if max_resize:
        for im in imgs:
            im.thumbnail((max_resize, max_resize))

    # user prompt（含占位 <image>），assistant 两个候选
    prompt_msgs = [{"role": "user", "content": [{"type": "image"}] + ([{"type": "text", "text": example["prompt"]}] if example.get("prompt") else [])}]
    chosen_msgs = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
    rejected_msgs = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]

    example["prompt"] = processor.apply_chat_template(prompt_msgs, tokenize=False)
    example["chosen"] = processor.apply_chat_template(chosen_msgs, tokenize=False)
    example["rejected"] = processor.apply_chat_template(rejected_msgs, tokenize=False)
    # TRL 期望列：images, prompt, chosen, rejected
    return example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset_name", type=str, default=None, help="HuggingFace hub 的数据名，或本地 json/jsonl/csv")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--data_files", type=str, default=None, help="本地文件路径(逗号分隔)")
    parser.add_argument("--image_root", type=str, default="", help="图像相对路径的根目录")
    parser.add_argument("--output_dir", type=str, default="./outputs-dpo-qwen25vl")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO 的温度超参")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_qlora", action="store_true", help="如需 4-bit QLoRA")
    parser.add_argument("--bf16", action="store_true", help="开启 BF16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, default="all-linear")
    parser.add_argument("--max_steps", type=int, default=-1, help="优先于 num_train_epochs")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    # Processor（既做 tokenizer 也做 image_processor）
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    # 模型 & (可选)QLoRA
    bnb_config = None
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        model = _QWEN_CLASS.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
    else:
        model = _QWEN_CLASS.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
        )

    # LoRA 仅训练 adapter
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 读取数据
    if args.dataset_name:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    else:
        if not args.data_files:
            raise ValueError("本地数据需提供 --data_files=path.jsonl 或多文件逗号分隔")
        files = [p.strip() for p in args.data_files.split(",")]
        ext = os.path.splitext(files[0])[-1].lower()
        if ext in [".json", ".jsonl"]:
            dataset = load_dataset("json", data_files=files, split="train")
        elif ext in [".csv", ".tsv"]:
            dataset = load_dataset("csv", data_files=files, split="train")
        else:
            raise ValueError(f"不支持的文件类型: {ext}")

    # 将路径转 PIL，标准化列
    dataset = dataset.map(lambda ex: ensure_images_in_memory(ex, args.image_root))

    # 按聊天模板串化
    dataset = dataset.map(lambda ex: format_for_chat(ex, processor), remove_columns=[
        c for c in dataset.column_names if c not in {"images", "prompt", "chosen", "rejected"}
    ])

    # 强制把 images 列 decode 为图像，避免 bytes
    feats = dataset.features
    if not isinstance(feats.get("images"), Sequence):
        feats["images"] = Sequence(HFImage(decode=True))
        dataset = dataset.cast(feats)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_grad_norm=args.max_grad_norm,
        report_to="none",
    )

    # DPO Trainer（VLM）
    trainer = DPOTrainer(
        model=model,
        ref_model=None,              # 使用 PEFT 时，ref_model 可为 None（内部会复制冻结）
        args=training_args,
        beta=args.beta,
        train_dataset=dataset,
        tokenizer=processor,         # 关键：传入 AutoProcessor (tokenizer + image_processor)
        peft_config=peft_config,
    )

    trainer.train()
    # 只保存 LoRA 适配器（默认行为）
    trainer.save_model()

    # 同步保存 processor（方便推理）
    processor.save_pretrained(args.output_dir)

    print("训练完成。LoRA 适配器已保存到:", args.output_dir)


if __name__ == "__main__":
    main()
