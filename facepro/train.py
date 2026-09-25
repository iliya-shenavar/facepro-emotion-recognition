from __future__ import annotations

import argparse
import math
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from facepro.data import get_dataloaders, train_split_class_counts
from facepro.models import build_model


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target, weight=self.weight, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def _class_weights_tensor(counts: list[int], mode: str, device: torch.device):
    if mode == "none":
        return None
    c = torch.tensor(counts, dtype=torch.float32, device=device).clamp(min=1.0)
    if mode == "balanced":
        w = c.sum() / (len(c) * c)
    elif mode == "sqrt":
        w = 1.0 / torch.sqrt(c)
    else:
        raise ValueError("class_weight_mode must be none|balanced|sqrt")
    w = w * (len(w) / w.sum())
    return w


def mixup_cutmix(x, y, num_classes, alpha=0.2, cutmix_alpha=1.0, p=0.5):
    if np.random.rand() > p:
        y_onehot = F.one_hot(y, num_classes).float()
        return x, y_onehot
    use_cutmix = np.random.rand() < 0.5
    lam = float(np.random.beta(cutmix_alpha if use_cutmix else alpha, cutmix_alpha if use_cutmix else alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    y_onehot = F.one_hot(y, num_classes).float()
    y2_onehot = F.one_hot(y[idx], num_classes).float()
    if use_cutmix:
        h, w = x.shape[2], x.shape[3]
        r = math.sqrt(1.0 - lam)
        cw, ch = int(w * r), int(h * r)
        cx, cy = np.random.randint(w), np.random.randint(h)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        x[:, :, y1:y2, x1:x2] = x[idx][:, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
        mixed_y = lam * y_onehot + (1.0 - lam) * y2_onehot
    else:
        x = lam * x + (1.0 - lam) * x[idx]
        mixed_y = lam * y_onehot + (1.0 - lam) * y2_onehot
    return x, mixed_y


def soft_ce(logits, target_probs, weight=None, label_smoothing=0.0):
    n = logits.size(-1)
    if label_smoothing > 0:
        target_probs = target_probs * (1 - label_smoothing) + label_smoothing / n
    logp = F.log_softmax(logits, dim=-1)
    loss = -(target_probs * logp).sum(dim=-1)
    if weight is not None:
        sample_w = (target_probs * weight.unsqueeze(0)).sum(dim=-1)
        loss = loss * sample_w
    return loss.mean()


def run_epoch(
    model,
    loader,
    criterion,
    device,
    train: bool,
    optimizer=None,
    scaler=None,
    num_classes: int = 7,
    use_mixup: bool = False,
    class_weight_tensor=None,
    label_smoothing: float = 0.0,
    grad_clip: float = 1.0,
):
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    amp_enabled = scaler is not None and device.type == "cuda"
    with ctx:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            if train and use_mixup:
                x, y_soft = mixup_cutmix(x, y, num_classes)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(x)
                    loss = soft_ce(logits, y_soft, weight=class_weight_tensor, label_smoothing=label_smoothing)
            else:
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(x)
                    loss = criterion(logits, y)
            if train:
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=2):
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    p = argparse.ArgumentParser(description="Train emotion classifier on faces")
    p.add_argument("--data_dir", type=str, default="data/fer2013/train")
    p.add_argument("--use_hf", action="store_true")
    p.add_argument("--model", type=str, default="rescnn",
                    choices=["small", "rescnn", "resnet18", "efficientnet_b0", "mobilenet_v3"])
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--subset_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--num_workers", type=int, default=0 if platform.system() == "Windows" else 4)
    p.add_argument("--class_weight_mode", type=str, default="balanced", choices=["none", "balanced", "sqrt"])
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--loss", type=str, default="focal", choices=["ce", "focal"])
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--mixup", action="store_true", default=True)
    p.add_argument("--no-mixup", dest="mixup", action="store_false")
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without val improvement)")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_names = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
        subset_ratio=args.subset_ratio,
        use_hf=args.use_hf,
    )
    num_classes = len(class_names)
    model = build_model(
        args.model, num_classes=num_classes, pretrained=args.pretrained, freeze_backbone=args.freeze_backbone
    ).to(device)

    if args.use_hf:
        tr_counts = [1] * num_classes
        cw_mode = "none"
        print("HF loader: per-class loss weights disabled (use local folders for balanced weights).")
    else:
        _, tr_counts = train_split_class_counts(args.data_dir, args.val_ratio, args.seed, args.subset_ratio)
        cw_mode = args.class_weight_mode
        print("Train samples per class:", dict(zip(class_names, tr_counts)))

    w = _class_weights_tensor(tr_counts, cw_mode, device)
    if w is not None:
        print("Class loss weights:", dict(zip(class_names, [round(float(x), 3) for x in w.detach().cpu().tolist()])))

    if args.loss == "focal":
        criterion = FocalLoss(weight=w, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        filter(lambda p_: p_.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4
    )
    scheduler = build_scheduler(optimizer, args.epochs, len(train_loader), args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    epochs_no_improve = 0

    print(f"Device: {device}  |  batches/epoch: {len(train_loader)}  |  batch_size: {args.batch_size}")

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"epoch {ep:03d}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp_enabled = args.amp and device.type == "cuda"
            if args.mixup:
                xb, y_soft = mixup_cutmix(x, y, num_classes)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(xb)
                    loss = soft_ce(logits, y_soft, weight=w, label_smoothing=args.label_smoothing)
            else:
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(x)
                    loss = criterion(logits, y)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss=f"{total_loss / max(1, total):.4f}", acc=f"{correct / max(1, total):.3f}")
        tr_loss, tr_acc = total_loss / max(1, total), correct / max(1, total)

        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, train=False, num_classes=num_classes)

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {ep:03d}/{args.epochs}  lr={cur_lr:.2e}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            epochs_no_improve = 0
            ckpt = {
                "model": args.model,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "img_size": args.img_size,
                "num_classes": num_classes,
            }
            torch.save(ckpt, out / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping: no val improvement in {args.patience} epochs.")
                break

    print(f"Best val accuracy: {best_acc:.3f}  checkpoint: {out / 'best.pt'}")


if __name__ == "__main__":
    main()
