from __future__ import annotations

import argparse

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _eval_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def pil_to_model_input(im: Image.Image, img_size: int) -> torch.Tensor:
    return _eval_transform(img_size)(im.convert("RGB")).unsqueeze(0)


def load_image(path: str, img_size: int):
    im = Image.open(path).convert("RGB")
    return pil_to_model_input(im, img_size)


def predict_probs(model, x: torch.Tensor, use_tta: bool = True) -> torch.Tensor:
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        if use_tta:
            flipped = torch.flip(x, dims=[3])
            probs_flip = torch.softmax(model(flipped), dim=1)
            probs = (probs + probs_flip) / 2.0
    return probs.squeeze(0)


def main():
    p = argparse.ArgumentParser(description="Predict emotion for one face image")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--tta", action="store_true", default=True)
    p.add_argument("--no-tta", dest="tta", action="store_false")
    args = p.parse_args()

    from facepro.models import build_model

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_names = ckpt["class_names"]
    img_size = int(ckpt.get("img_size", 48))
    model = build_model(ckpt["model"], num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = load_image(args.image, img_size)
    probs = predict_probs(model, x, use_tta=args.tta)
    topk = probs.topk(min(5, len(class_names)))
    print("Top predictions:")
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"  {class_names[idx]}: {score:.3f}")


if __name__ == "__main__":
    main()
