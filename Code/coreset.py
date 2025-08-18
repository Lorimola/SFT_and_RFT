import json
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

# use clip to take embedding
MODEL_PATH = "/data2/home/donglingzhong/yangsb/SAR/models/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_PATH)


def encode_element(image_paths, batch_size=32):
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        images = []
        for p in batch:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"No picture {p} with error {e}.")
        if not images:
            continue
        # put images into model to take embedding
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
        embeddings.append(emb.cpu().numpy())
    if len(embeddings) == 0:
        return None
    # merge nparrays in vertical axis
    return np.mean(np.vstack(embeddings), axis=0)


# calculate KL Divergence
def kl_divergence(p, q, eps=1e-10):
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return np.sum(p * np.log((p + eps) / (q + eps)))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# imagr_path in android control is a list, but in AITZ is a string
def count_images(data, is_control=False):
    if is_control:
        return sum(len(item["image_path"]) for item in data)
    else:
        return len(data)


def select_subset(control_data, target_img_count):
    """
    Select subset from android control such that the number of images in subset 
    is equal to AITZ.
    """
    # calculate the embedding of an element in android control
    element_embeddings = []
    valid_indices = []
    for si, sample in enumerate(tqdm(control_data, desc="Encoding elements")):
        emb = encode_element(sample["image_path"])
        if emb is not None:
            element_embeddings.append(emb)
            valid_indices.append(si)

    if len(element_embeddings) == 0:
        raise ValueError("No valid embeddings!")

    element_embeddings = np.array(element_embeddings)

    # calculate KL Divergence to match distribution
    k = min(50, len(element_embeddings) // 2 if len(element_embeddings) > 1 else 1)
    km = KMeans(n_clusters=k, random_state=0).fit(element_embeddings)
    labels = km.labels_
    full_hist = np.bincount(labels, minlength=k).astype(float)

    selected_samples = set()
    selected_hist = np.zeros(k, dtype=int)
    total_imgs = 0

    while total_imgs < target_img_count:
        best_si, best_score = None, float("inf")
        for idx, si in enumerate(valid_indices):
            if si in selected_samples:
                continue
            tmp_hist = selected_hist.copy()
            tmp_imgs = 0
            elem_label = labels[idx]
            tmp_hist[elem_label] += 1
            tmp_imgs += len(control_data[si]["image_path"])
            score = kl_divergence(full_hist, tmp_hist)
            # use greedy algorithm to get best sample
            if score < best_score:
                best_score = score
                best_si = si
                best_elem_label = elem_label
                best_imgs = tmp_imgs

        selected_samples.add(best_si)
        selected_hist[best_elem_label] += 1
        total_imgs += best_imgs

    return [control_data[si] for si in selected_samples]


def main(aitz_train, aitz_test, control_train, control_test):
    aitz_train_data = load_json(aitz_train)
    aitz_test_data = load_json(aitz_test)
    control_train_data = load_json(control_train)
    control_test_data = load_json(control_test)

    target_train_imgs = count_images(aitz_train_data, is_control=False)
    target_test_imgs = count_images(aitz_test_data, is_control=False)

    print(f"AITZ train: {target_train_imgs}")
    print(f"AITZ test: {target_test_imgs}")

    subset_train = select_subset(control_train_data, target_train_imgs)
    subset_test = select_subset(control_test_data, target_test_imgs)

    with open("/data2/home/donglingzhong/yangsb/SAR/Dateset/android_control/train_subset.json", "w", encoding="utf-8") as f:
        json.dump(subset_train, f, indent=2, ensure_ascii=False)
    with open("/data2/home/donglingzhong/yangsb/SAR/Dateset/android_control/test_subset.json", "w", encoding="utf-8") as f:
        json.dump(subset_test, f, indent=2, ensure_ascii=False)

    print("Finish!")


if __name__ == "__main__":
    main(
        "/data2/home/donglingzhong/yangsb/SAR/Dateset/AITZ/train.json",
        "/data2/home/donglingzhong/yangsb/SAR/Dateset/AITZ/test.json",
        "/data2/home/donglingzhong/yangsb/SAR/Dateset/android_control/train.json",
        "/data2/home/donglingzhong/yangsb/SAR/Dateset/android_control/test.json"
    )

