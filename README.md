# The Difference Between SFT and RFT

> This experiment focuses on the role of SFT and RFT in improving the performance of the model.

---

## Experimental Design
The model selected for this expriment is `Qwen2.5VL-7B-Instruct`.

### Expriment Based on AITZ and Android Control
The datasets selected for this expriment is `AITZ` and `Android Control`, which are used for cross-training. The basic idea behind this experiment is that when a model is trained using one type of dataset, testing with the same type of data is considered `ID`, while testing with another type of dataset is considered `OOD`. By comparing the differences in performance between SFT and RFT-trained models on ID and OOD datasets, we can determine the differences in the contributions of SFT and RFT to model performance improvement.

#### Dataset Filtering
The size of `Android Control` is far larger than `AITZ`, so we should filter dataset to ensure that the training set and test set sizes for the two types of datasets are equal. The criterion for evaluating the size of a dataset is the number of images. Inspired by Coreset's view, we decided to calculate the `embedding distribution` of the original dataset, and then use a `greedy algorithm` to select the subset with the smallest `KL divergence` that is closest to the distribution of the original dataset.

#### SFT and RFT Training

---
