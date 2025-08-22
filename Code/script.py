import os
import json
import argparse
from typing import Dict, Any, List, Optional

from datasets import Dataset


# Turn any type into a string
def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


# Convert AITZ sample
def convert_aitz_record(ex: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    img = ex.get("image_path") or ex.get("image_full_path")
    if not img:
        return None

    # Build user content
    user_parts: List[str] = []
    if ex.get("task"):
        user_parts.append(f"Task: {ex['task']}")
    if ex.get("ui_positions"):
        user_parts.append(f"UI Positions: {_as_text(ex['ui_positions'])}")
    if ex.get("ui_text"):
        user_parts.append(f"UI Text: {_as_text(ex['ui_text'])}")
    if ex.get("ui_types"):
        user_parts.append(f"UI Types: {_as_text(ex['ui_types'])}")
    if ex.get("coat_screen_desc"):
        user_parts.append(f"Screen Description: {_as_text(ex['coat_screen_desc'])}")

    user_content = [{"type": "image", "image": f"file://{img}"}]
    if user_parts:
        user_content.append({"type": "text", "text": "\n".join(user_parts)})

    # Build assistant content
    asst_parts: List[str] = []
    if ex.get("result_action_type"):
        asst_parts.append(str(ex["result_action_type"]))
    if ex.get("result_action_text"):
        asst_parts.append(_as_text(ex["result_action_text"]))
    if ex.get("result_touch_yx"):
        asst_parts.append(f"Touch yx: {_as_text(ex['result_touch_yx'])}")
    if ex.get("result_lift_yx"):
        asst_parts.append(f"Lift yx: {_as_text(ex['result_lift_yx'])}")
    if ex.get("coat_action_think"):
        asst_parts.append(f"Action think: {_as_text(ex['coat_action_think'])}")
    if ex.get("coat_action_desc"):
        asst_parts.append(_as_text(ex["coat_action_desc"]))
    if ex.get("coat_action_result"):
        asst_parts.append(f"Action result: {_as_text(ex['coat_action_result'])}")

    if not asst_parts:
        return None

    assistant_content = [{"type": "text", "text": "\n".join(asst_parts)}]

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    # Build DPO fields
    prompt_text = "\n".join(user_parts).strip()
    chosen_text = "\n".join(asst_parts).strip()
    rejected_text = chosen_text[: max(1, len(chosen_text) // 2)] if chosen_text else "I cannot answer."

    return {
        "id": str(ex.get("episode_id", "")) + "_" + str(ex.get("step_id", "")),
        "messages": messages,
        "prompt_text": prompt_text,
        "chosen_text": chosen_text,
        "rejected_text": rejected_text,
    }


# Convert control episode
def convert_control_episode(ep: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_paths: List[str] = ep.get("image_path") or []
    actions: List[str] = ep.get("actions") or []
    tasks: List[str] = ep.get("task") or []
    node_infos: List[List[Dict[str, Any]]] = ep.get("node_info") or []
    widths: List[int] = ep.get("screenshot_widths") or []
    heights: List[int] = ep.get("screenshot_heights") or []
    goal: str = (ep.get("goal") or "").strip()
    accessibilities: List[str] = ep.get("accessibility_tree") or []

    # User content with images
    user_content = []
    for i, img in enumerate(image_paths):
        img_info = {
            "type": "image",
            "image": f"file://{img}",
            "resized_height": heights[i] if i < len(heights) else None,
            "resized_width": widths[i] if i < len(widths) else None,
            "accessibility": accessibilities[i] if i < len(accessibilities) else None,
        }
        user_content.append(img_info)

    # Add text-based information
    text_parts = []
    if goal:
        text_parts.append(f"Goal: {goal}")

    for i, task in enumerate(tasks):
        if task:
            text_parts.append(f"Task {i}: {task}")

    for i, node_info in enumerate(node_infos):
        text_parts.append(f"Node info {i}: {node_info}")

    if text_parts:
        user_content.append({"type": "text", "text": "\n".join(text_parts)})

    # Assistant answer
    ans = "\n".join([str(a).strip() for a in actions if a])
    assistant_content = [{"type": "text", "text": ans}]

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    # Build DPO fields
    prompt_text = "\n".join(text_parts).strip()
    chosen_text = ans
    rejected_text = chosen_text[: max(1, len(chosen_text) // 2)] if chosen_text else "I cannot answer."

    sid = str(ep.get("episode_id", ep.get("index", "ep")))
    return [{
        "id": sid,
        "messages": messages,
        "prompt_text": prompt_text,
        "chosen_text": chosen_text,
        "rejected_text": rejected_text,
    }]


def main():
    # Pass parameters by command line
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", type=str, required=True, help="input json file path")
    ap.add_argument("--output_dir", type=str, required=True, help="output directory")
    ap.add_argument("--dataset_type", type=str, choices=["aitz", "control"], required=True,
                    help="dataset type: aitz or control")
    args = ap.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    all_samples: List[Dict[str, Any]] = []
    if args.dataset_type == "aitz":
        for ex in raw_list:
            item = convert_aitz_record(ex)
            if item:
                all_samples.append(item)
    else:
        for ex in raw_list:
            items = convert_control_episode(ex)
            all_samples.extend(items)

    ds = Dataset.from_list(all_samples)
    os.makedirs(args.output_dir, exist_ok=True)
    ds.save_to_disk(args.output_dir)
    print(f"Saved {len(ds)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
