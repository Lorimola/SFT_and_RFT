import json, os
from typing import Dict, Any, List

def to_llava_sample_android(sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """
    Convert one Android Control sample to LLaVA/Qwen format,
    while preserving all original fields.
    """
    image = sample.get("image_path")
    assert image, "Android sample must include image_path"

    instruction = sample.get("instruction", "请根据屏幕完成下一步操作")

    gold = sample.get("target_action") or sample.get("answer") or ""

    # conversations
    conversations = [
        {"from": "human", "value": f"<image>\n {instruction}"},
        {"from": "gpt", "value": str(gold).strip()}
    ]

    out_sample = dict(sample)
    out_sample.update({
        "id": f"{dataset_name}_{sample.get('id','unk')}_{sample.get('step_id','0')}",
        "image": image,
        "conversations": conversations,
    })

    return out_sample


def to_llava_sample_aitz(sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    image = sample.get("image_path") or sample.get("image_full_path")
    assert image, "AITZ sample must include image_path"

    task = sample.get("task", "请根据屏幕完成下一步操作")
    gold = sample.get("coat_action_desc") or sample.get("result_action_text") or ""

    conversations = [
        {"from": "human", "value": f"<image>\n {task}"},
        {"from": "gpt", "value": str(gold).strip()}
    ]

    out_sample = dict(sample)
    out_sample.update({
        "id": f"{dataset_name}_{sample.get('episode_id','unk')}_{sample.get('step_id','0')}",
        "image": image,
        "conversations": conversations,
    })

    return out_sample


def convert(in_json_path: str, out_json_path: str, dataset_name: str, dataset_type: str):
    data: List[Dict[str, Any]] = []
    with open(in_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    for s in raw:
        try:
            if dataset_type == "AITZ":
                data.append(to_llava_sample_aitz(s, dataset_name))
            elif dataset_type == "AndroidCtrl":
                data.append(to_llava_sample_android(s, dataset_name))
        except AssertionError:
            continue

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} samples to {out_json_path}")


if __name__ == "__main__":
    convert(
        in_json_path="/data2/home/donglingzhong/yangsb/Dateset/AITZ/origin_train.json",
        out_json_path="/data2/home/donglingzhong/yangsb/Dateset/AITZ/train.json",
        dataset_name="AITZ",
        dataset_type="AITZ"
    )
    convert(
        in_json_path="/data2/home/donglingzhong/yangsb/Dateset/AndroidCtrl/origin_train.json",
        out_json_path="/data2/home/donglingzhong/yangsb/Dateset/AndroidCtrl/train.json",
        dataset_name="AndroidCtrl",
        dataset_type="AndroidCtrl"
    )
