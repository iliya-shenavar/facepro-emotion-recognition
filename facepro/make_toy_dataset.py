from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/fer2013/train")
    p.add_argument("--per_class", type=int, default=40)
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    for ci, c in enumerate(classes):
        d = root / c
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):

            base = (ci * 18) % 256
            noise = rng.integers(0, 80, size=(args.size, args.size), dtype=np.uint8)
            arr = np.clip(base + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(d / f"{i:04d}.png")
    print(f"Wrote {args.per_class} images x {len(classes)} classes under {root}")


if __name__ == "__main__":
    main()
