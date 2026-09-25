from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from facepro.data import get_dataloaders
from facepro.models import build_model
from facepro.train import run_epoch


def load_checkpoint(path: str):
    kw = {"map_location": "cpu"}
    try:
        kw["weights_only"] = False
        return torch.load(path, **kw)
    except TypeError:
        return torch.load(path, map_location="cpu")


def predict_with_tta(model, x, use_tta: bool):
    logits = model(x)
    if not use_tta:
        return logits
    flipped = torch.flip(x, dims=[3])
    logits_flip = model(flipped)
    return (torch.softmax(logits, dim=1) + torch.softmax(logits_flip, dim=1)) / 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--data_dir", type=str, default="data/fer2013/train")
    p.add_argument("--use_hf", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--tta", action="store_true", default=True)
    p.add_argument("--no-tta", dest="tta", action="store_false")
    args = p.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    class_names = ckpt["class_names"]
    img_size = int(ckpt.get("img_size", 48))
    model_name = ckpt["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, val_loader, names2 = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        img_size=img_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_hf=args.use_hf,
    )
    if names2 != class_names:
        print("Warning: class order may differ from training; report labels use checkpoint order.")

    criterion = nn.CrossEntropyLoss()
    _, acc = run_epoch(model, val_loader, criterion, device, train=False, num_classes=len(class_names))
    print(f"Val accuracy (no TTA): {acc:.4f}")

    y_true, y_pred = [], []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            probs = predict_with_tta(model, x, args.tta)
            pred = probs.argmax(dim=1).cpu()
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
            correct += (pred == y).sum().item()
            total += y.size(0)

    if args.tta:
        print(f"Val accuracy (with TTA): {correct / max(1, total):.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
