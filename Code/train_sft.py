from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import os
import random

def build_prompt_and_response(example):
    """
    构造prompt和response，适配AITZ和android_control两类数据。
    """
    img_field = example.get("image_path", None)
    # image_path 可能是str或list
    if isinstance(img_field, str):
        # AITZ类型样本
        prompt_parts = []
        prompt_parts.append(f"ImagePath: {img_field}")
        
        # task
        if "task" in example and example["task"]:
            prompt_parts.append(f"Task: {example['task']}")
        
        # 屏幕描述
        if "coat_screen_desc" in example and example["coat_screen_desc"]:
            prompt_parts.append(f"ScreenDescription: {example['coat_screen_desc']}")
        
        # 动作思考
        if "coat_action_think" in example and example["coat_action_think"]:
            prompt_parts.append(f"ActionThink: {example['coat_action_think']}")
        
        # 动作描述
        if "coat_action_desc" in example and example["coat_action_desc"]:
            prompt_parts.append(f"ActionDesc: {example['coat_action_desc']}")
        
        prompt = "\n".join(prompt_parts)
        
        # 期望模型输出动作结果
        response = example.get("coat_action_result", None)
        if response is None or response == "":
            # 找不到动作结果时，尝试用其他可能字段或跳过
            response = example.get("response", None)
        if response is None or response == "":
            return None, None
        
        return prompt.strip(), str(response).strip()

    elif isinstance(img_field, list):
        # android_control类型样本
        prompt_parts = []
        prompt_parts.append("ImagePaths:")
        prompt_parts.extend(img_field)  # 保留全部图片路径

        # 如果有任务字段
        if "task" in example and example["task"]:
            prompt_parts.append(f"Task: {example['task']}")

        prompt = "\n".join(prompt_parts)

        # 响应字段按常规
        for resp_key in ("response", "output", "gt", "answer", "label", "target"):
            if resp_key in example and example[resp_key] not in (None, ""):
                return prompt.strip(), str(example[resp_key]).strip()

        # 找不到有效响应，跳过
        return None, None

    else:
        # 不符合预期的结构，跳过
        return None, None


def preprocess_function(example, tokenizer, max_length=2048):
    """
    map 用的逐样本 preprocess：构建 prompt+response 并 token 化。
    返回 token ids 与 labels（labels 与 input_ids 相同，用于 SFT）。
    """
    prompt, response = build_prompt_and_response(example)
    if prompt is None or response is None:
        # 标记为 None，让 map 跳过（datasets.map 里处理）
        return None

    # 你可以把提示模板自由调整 — 下面是一个标准对话式 SFT 模板
    full_input = f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
    # 编码
    tokenized = tokenizer(full_input, truncation=True, max_length=max_length, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def train_sft(model_name="/data1/models/Qwen2.5-VL-7B-Instruct",
              train_json="/path/to/your/train.json",
              output_dir="/data2/home/donglingzhong/yangsb/SAR/models/SFT",
              per_device_train_batch_size=2,
              gradient_accumulation_steps=8,
              num_train_epochs=3,
              max_length=2048,
              seed=42):
    """
    主训练函数
    - train_json: 你会传入完整路径（不要改这里的变量名）
    - 会逐元素地构造 prompt/response（保留 android_control 的 image_path 列表完整性）
    """

    # 设置随机种子以便重复性
    random.seed(seed)
    torch.manual_seed(seed)

    # 加载 tokenizer 与 model（保持你给的默认路径）
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    # 加载 json（注意：将传入的完整路径作为 train 数据）
    assert os.path.exists(train_json), f"train_json 路径不存在: {train_json}"
    dataset = load_dataset("json", data_files={"train": train_json}, field=None)

    # 逐元素 map（非批处理），并跳过返回 None 的样本
    def _map_fn(example):
        res = preprocess_function(example, tokenizer, max_length=max_length)
        return res if res is not None else {}

    tokenized = dataset["train"].map(_map_fn, remove_columns=dataset["train"].column_names, batched=False)

    # datasets.map 返回空 dict 的样本可能存在，需要过滤掉（labels 为 [] 或 input_ids 缺失的）
    def filter_valid_examples(example):
        # 有效的样本必须含有 input_ids 且长度>0
        return "input_ids" in example and example["input_ids"] and "labels" in example and example["labels"]

    tokenized = tokenized.filter(filter_valid_examples)

    # TrainingArguments（保留你原来的训练超参，可以按需调整）
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=True,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        output_dir=output_dir,
        save_total_limit=2,
        evaluation_strategy="no",
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(output_dir)
    print("训练完成，模型已保存到：", output_dir)


if __name__ == "__main__":
    # 请在调用时把 train_json 改成你真实传入的完整路径，例如：
    # train_sft(train_json="/data2/you/path/android_control/train.json")
    train_sft()

