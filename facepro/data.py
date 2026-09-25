from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _train_transforms(img_size: int, use_augment: bool):
    if not use_augment:
        return _eval_transforms(img_size)
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )


def _eval_transforms(img_size: int):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _shuffle_split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    va = sorted(idx[-n_val:])
    tr = sorted(idx[:-n_val])
    return tr, va


def _try_hf_loaders(
    batch_size: int,
    img_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int,
    use_augment: bool,
):
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    raw = load_dataset("blanchon/FER2013")
    hf_train = raw["train"]
    feat = hf_train.features.get("labels")
    class_names = list(feat.names) if feat is not None and hasattr(feat, "names") else [str(i) for i in range(7)]

    class IdxDS(torch.utils.data.Dataset):
        def __init__(self, base, idx_list, eval_mode: bool):
            self.base = base
            self.idx_list = idx_list
            self.eval_mode = eval_mode

        def __len__(self):
            return len(self.idx_list)

        def __getitem__(self, i):
            j = int(self.idx_list[i])
            row = self.base[j]
            im = row["image"]
            if getattr(im, "mode", "") != "RGB":
                im = im.convert("RGB")
            lab = row.get("labels", row.get("label", 0))
            tt = _eval_transforms(img_size) if self.eval_mode else _train_transforms(img_size, use_augment)
            return tt(im), int(lab)

    n = len(hf_train)
    tr_idx, va_idx = _shuffle_split_indices(n, val_ratio, seed)

    tr_ds = IdxDS(hf_train, tr_idx, eval_mode=False)
    va_ds = IdxDS(hf_train, va_idx, eval_mode=True)

    train_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        va_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, class_names


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    img_size: int = 48,
    num_workers: int = 0,
    seed: int = 42,
    subset_ratio: float = 1.0,
    use_augment: bool = True,
    use_hf: bool = False,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    if use_hf:
        out = _try_hf_loaders(batch_size, img_size, val_ratio, seed, num_workers, use_augment)
        if out is not None:
            return out

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Place FER2013-style folders (one per class) here, or install `datasets` and pass --use_hf."
        )

    train_tf = _train_transforms(img_size, use_augment)
    eval_tf = _eval_transforms(img_size)

    full_train = datasets.ImageFolder(str(data_dir), transform=train_tf)
    full_eval = datasets.ImageFolder(str(data_dir), transform=eval_tf)
    class_names = full_train.classes

    n = len(full_train)
    tr_idx, va_idx = _shuffle_split_indices(n, val_ratio, seed)

    if subset_ratio < 1.0:
        rng = random.Random(seed)
        k = max(1, int(len(tr_idx) * subset_ratio))
        tr_idx = sorted(rng.sample(tr_idx, k=k))

    train_ds = Subset(full_train, tr_idx)
    val_ds = Subset(full_eval, va_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_ds) > batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, class_names


def train_split_class_counts(
    data_dir: str | Path,
    val_ratio: float,
    seed: int,
    subset_ratio: float,
) -> Tuple[list[str], list[int]]:
    data_dir = Path(data_dir)
    full = datasets.ImageFolder(str(data_dir))
    n = len(full)
    tr_idx, _ = _shuffle_split_indices(n, val_ratio, seed)
    if subset_ratio < 1.0:
        rng = random.Random(seed)
        k = max(1, int(len(tr_idx) * subset_ratio))
        tr_idx = sorted(rng.sample(tr_idx, k=k))
    counts = [0] * len(full.classes)
    for i in tr_idx:
        _, y = full[i]
        counts[y] += 1
    return full.classes, counts
