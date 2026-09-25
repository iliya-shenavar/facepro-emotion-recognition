<div align="center">

# 🎭 FacePro

### Facial Emotion Recognition Toolkit

**A PyTorch toolkit for training and running facial emotion recognition models on FER2013-style data.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%9D%A4-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](#-license)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)]()

*CNNs · SE-ResNet · Transfer Learning · HOG+LBP+SVM · Real-time Webcam Demo*

</div>

---

## 📖 Overview

**FacePro** ships several trainable architectures — a lightweight CNN, a custom SE-ResNet, and transfer-learning heads for ResNet18 / EfficientNet-B0 / MobileNetV3 — plus a classical **HOG + LBP + SVM** baseline for comparison, and a real-time **webcam demo** that detects faces and overlays the predicted emotion live.

---

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### 🧠 Multiple Model Backbones
Pick per training run via `--model`:

| Key | Description |
|-----|-------------|
| `small` | Compact CNN baseline |
| `rescnn` | Custom residual CNN with **Squeeze-and-Excitation** blocks (`EmotionResCNN`) |
| `resnet18` | ImageNet-pretrained torchvision backbone |
| `efficientnet_b0` | ImageNet-pretrained torchvision backbone |
| `mobilenet_v3` | ImageNet-pretrained torchvision backbone |

All transfer models support a **fine-tunable head** and optional **backbone freezing**.

</td>
<td width="50%" valign="top">

### 🚀 Robust Training Loop
- 🎯 Focal Loss or label-smoothed Cross-Entropy
- ⚖️ Per-class loss weighting (`balanced` / `sqrt`)
- 🔀 MixUp / CutMix
- 🎨 RandAugment
- 📉 Cosine LR schedule with warmup
- ✂️ Gradient clipping
- ⚡ AMP mixed precision
- 🛑 Early stopping

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📊 Evaluation with TTA
- 🔄 Horizontal-flip test-time augmentation
- 📋 Per-class classification report
- 🧩 Confusion matrix

</td>
<td width="50%" valign="top">

### 🎥 Inference & Live Demo
- 🖼️ Single-image prediction
- 📹 Live webcam demo with:
  - Haar-cascade face detection
  - CLAHE contrast enhancement
  - Temporal smoothing
  - Majority-vote stabilization

</td>
</tr>
<tr>
<td colspan="2" valign="top">

### 🗂️ Data Preparation Utilities
- 📥 Build `ImageFolder` dataset from the classic **FER2013 CSV**
- 🤗 Build from the `blanchon/FER2013` **Hugging Face** dataset
- 🧪 Generate a **synthetic toy dataset** for smoke-testing the pipeline

</td>
</tr>
</table>

---

## 🏗️ Project Structure

```text
Facepro/
├── facepro/
│   ├── config.py                      # Canonical label names
│   ├── data.py                        # Dataloaders, transforms, train/val split
│   ├── models.py                      # All model architectures
│   ├── train.py                       # Training loop (loss, mixup, scheduler, checkpointing)
│   ├── evaluate.py                    # Checkpoint evaluation + TTA + metrics
│   ├── predict.py                     # Single-image inference
│   ├── live.py                        # Real-time webcam demo
│   ├── classical.py                   # HOG/LBP + SVM baseline
│   ├── import_fer2013_to_folders.py   # fer2013.csv -> ImageFolder layout
│   ├── download_hf_fer_to_folders.py  # HF `blanchon/FER2013` -> ImageFolder layout
│   └── make_toy_dataset.py            # Tiny synthetic dataset for testing
├── data/
│   └── fer2013/
│       └── train/                     # ImageFolder root (empty — see "Dataset")
│           ├── angry/
│           ├── disgust/
│           ├── fear/
│           ├── happy/
│           ├── neutral/
│           ├── sad/
│           └── surprise/
├── checkpoints/                       # Trained models saved here (gitignored)
└── requirements.txt
```

---

## ⚙️ Installation

> **Requires Python 3.10+** (developed / tested on 3.11)

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

<details>
<summary><b>🤗 Optional: Hugging Face support</b> (only needed for <code>--use_hf</code>)</summary>

```bash
pip install datasets
```

</details>

---

## 📦 Dataset

> [!IMPORTANT]
> The `data/fer2013/train/<class>/` folders in this repository are **empty**.
> The original training images were removed before pushing to GitHub (to keep the repository small / avoid uploading tens of thousands of files).
> **You must populate them yourself before training or evaluating.**
> Only an empty `.gitkeep` file is kept per folder to preserve the directory structure.

`data/fer2013/train/` must end up as an **`ImageFolder`-compatible layout** — one subfolder per class:

```text
data/fer2013/train/
├── angry/*.png
├── disgust/*.png
├── fear/*.png
├── happy/*.png
├── neutral/*.png
├── sad/*.png
└── surprise/*.png
```

### 🔽 Pick **one** of the following ways to fill it:

<table>
<tr>
<th align="left">Option</th>
<th align="left">Source</th>
<th align="left">Command</th>
</tr>
<tr>
<td><b>A</b></td>
<td>Classic <b>FER2013 CSV</b><br/><sub>(Kaggle <code>msambare/fer2013</code> or the original ICML challenge CSV)</sub></td>
<td>

```bash
python -m facepro.import_fer2013_to_folders \
    --csv path/to/fer2013.csv --usage Training
```

</td>
</tr>
<tr>
<td><b>B</b></td>
<td><b>Hugging Face</b> dataset<br/><sub>(<code>blanchon/FER2013</code>, downloaded automatically)</sub></td>
<td>

```bash
pip install datasets
python -m facepro.download_hf_fer_to_folders
```

</td>
</tr>
<tr>
<td><b>C</b></td>
<td><b>Tiny synthetic toy dataset</b><br/><sub>(just to sanity-check the pipeline end-to-end — not for real training)</sub></td>
<td>

```bash
python -m facepro.make_toy_dataset --per_class 40
```

</td>
</tr>
</table>

> 💡 You can also **skip local files entirely** and stream from Hugging Face at training time with the `--use_hf` flag.

---

## 🚀 Usage

### 🏋️ Train

```bash
python -m facepro.train --model rescnn --epochs 40 --batch_size 128 --img_size 64
```

<details>
<summary><b>📝 Useful flags</b></summary>

| Flag | Values / Notes |
|------|----------------|
| `--model` | `small` · `rescnn` · `resnet18` · `efficientnet_b0` · `mobilenet_v3` |
| `--use_hf` | Stream from Hugging Face |
| `--loss` | `focal` · `ce` |
| `--class_weight_mode` | `none` · `balanced` · `sqrt` |
| `--mixup` / `--no-mixup` | Enable / disable MixUp & CutMix |
| `--pretrained` / `--no-pretrained` | ImageNet weights toggle |
| `--freeze_backbone` | Freeze feature extractor |
| `--patience` | Early-stopping patience |

Run `python -m facepro.train --help` for the full list.

</details>

> 🏆 The best checkpoint (by validation accuracy) is written to **`checkpoints/best.pt`**.

---

### 📊 Evaluate

```bash
python -m facepro.evaluate --checkpoint checkpoints/best.pt
```

Prints:
- ✅ Validation accuracy (with and without TTA)
- 📋 Per-class classification report
- 🧩 Confusion matrix

---

### 🖼️ Predict on a Single Image

```bash
python -m facepro.predict --checkpoint checkpoints/best.pt --image path/to/face.jpg
```

---

### 📹 Live Webcam Demo

```bash
python -m facepro.live --checkpoint checkpoints/best.pt
```

**How it works:**

1. 🔍 Detects the **largest face** via a Haar cascade
2. ✂️ Expands the crop with a margin
3. 🎨 Optionally applies **CLAHE** contrast enhancement
4. 🏷️ Overlays the **smoothed / majority-voted** predicted emotion on the video feed

> Press **`q`** to quit. See `--help` for camera index, backend, and smoothing options.

---

### 🧮 Classical Baseline (HOG + LBP + SVM)

```bash
python -m facepro.classical --data_dir data/fer2013/train
```

---

## 📝 Notes

- 🚫 `checkpoints/` and `*.pt` files are **gitignored** — trained weights are **not** included in this repository and must be produced locally by running `train.py`.
- 🪟 Training was developed with **Windows** in mind (`--num_workers` defaults to `0` on Windows; camera backends `msmf` / `dshow` for `live.py`), but all scripts run on **Linux / macOS** as well.

---

## 📜 License

Add a license of your choice (e.g. **MIT**) before publishing.

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ using PyTorch

</div>
