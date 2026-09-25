from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "fer2013" / "train"),
        help="ImageFolder root (one subfolder per class)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="HF split name (usually train; some mirrors also have test)",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing images under --out class folders before writing",
    )
    p.add_argument("--max_samples", type=int, default=0, help="0 = all samples in split")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install datasets first: pip install datasets") from e

    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("blanchon/FER2013")
    if args.split not in ds:
        raise SystemExit(f"Split '{args.split}' not in dataset. Available: {list(ds.keys())}")

    hf_split = ds[args.split]
    lab_feat = hf_split.features["labels"]
    names = getattr(lab_feat, "names", None)
    if isinstance(names, dict):
        id2name = {int(k): v for k, v in names.items()}
    elif isinstance(names, (list, tuple)):
        id2name = dict(enumerate(names))
    else:
        raise SystemExit("Unexpected labels feature; cannot resolve class names")

    if args.clean:
        for sd in sorted(root.iterdir()):
            if sd.is_dir():
                for f in sd.iterdir():
                    if f.is_file():
                        f.unlink()

    n = len(hf_split)
    if args.max_samples:
        n = min(n, args.max_samples)

    counts: dict[int, int] = {}
    for idx in tqdm(range(n), desc="Writing images"):
        row = hf_split[idx]
        im = row["image"]
        y = row["labels"] if "labels" in row else row["label"]
        yi = int(y)
        cls = str(id2name[yi]).lower()
        counts[yi] = counts.get(yi, 0) + 1
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        if getattr(im, "mode", "") != "L":
            im = im.convert("L")
        out_path = d / f"{counts[yi]:06d}.png"
        im.save(out_path)

    total = sum(counts.values())
    print(f"Done. Wrote {total} images under {root}")
    for yi in sorted(counts.keys()):
        print(f"  {id2name[yi]}: {counts[yi]}")


if __name__ == "__main__":
    main()
