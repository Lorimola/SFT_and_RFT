import json
import random

def count_images(json_path):
    """统计 json 文件中图片数量"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data[0], dict) and "image_path" in data[0]:
        # android_control 任务级
        return sum(len(task["image_path"]) for task in data)
    else:
        # AITZ 单步级
        return len(data)

def sample_to_match(json_path, target_img_count, output_path):
    """从 android_control 抽样，使图片数尽量接近 target_img_count，但不截断任务"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.shuffle(data)  # 打乱任务顺序

    selected_tasks = []
    total_images = 0

    for task in data:
        task_imgs = len(task["image_path"])
        if total_images + task_imgs > target_img_count:
            # 如果加上这个任务会超过目标，就结束
            break
        selected_tasks.append(task)
        total_images += task_imgs

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_tasks, f, ensure_ascii=False, indent=2)

    print(f"[{output_path}] 共采样 {total_images} 张图片，目标 {target_img_count} 张")



sample_to_match("/data2/home/donglingzhong/yangsb/SAR/Dateset/android_control/train.json", 13919,
                "/data2/home/donglingzhong/yangsb/SAR/Dateset/android_control/train_sampled.json")

sample_to_match("/data2/home/donglingzhong/yangsb/SAR/Dateset/android_control/test.json", 4724,
                "/data2/home/donglingzhong/yangsb/SAR/Dateset/android_control/test_sampled.json")

