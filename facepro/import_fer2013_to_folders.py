from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


INT2FOLDER = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def _normalize_fieldnames(fieldnames: list[str]) -> dict[str, str]:
    return {k.strip().lower(): k for k in fieldnames}


def import_from_csv(csv_path: Path, out_root: Path, usage_filter: str | None) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {v: 0 for v in INT2FOLDER.values()}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        norm = _normalize_fieldnames(reader.fieldnames or [])

        def col(*candidates: str) -> str | None:
            for c in candidates:
                k = norm.get(c.lower())
                if k:
                    return k
            return None

        k_emotion = col("emotion")
        k_pixels = col("pixels")
        k_usage = col("usage")

        if not k_emotion or not k_pixels:
            raise SystemExit(
                f"CSV must contain emotion + pixels columns. Got: {list(reader.fieldnames)}"
            )

        for row in reader:
            usage = row.get(k_usage, "").strip() if k_usage else ""
            if usage_filter and usage_filter != "all":
                if usage.lower() != usage_filter.lower():
                    continue
            y = int(row[k_emotion])
            folder = INT2FOLDER[y]
            pixels = row[k_pixels]
            arr = np.asarray(pixels.split(), dtype=np.uint8)
            if arr.size != 48 * 48:
                raise ValueError(f"Bad pixel row (expected 2304 ints), got {arr.size}")
            img = Image.fromarray(arr.reshape(48, 48), mode="L")
            counts[folder] += 1
            d = out_root / folder
            d.mkdir(parents=True, exist_ok=True)
            img.save(d / f"{counts[folder]:06d}.png")

    total = sum(counts.values())
    print(f"Wrote {total} PNG files under {out_root}")
    for name in sorted(INT2FOLDER.values()):
        print(f"  {name}: {counts[name]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="Path to fer2013.csv (if set, used directly)",
    )
    p.add_argument(
        "--csv_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "fer2013_source"),
        help="Root containing subfolder fer2013/fer2013.csv (torchvision-style layout)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "fer2013" / "train"),
        help="ImageFolder output root",
    )
    p.add_argument(
        "--usage",
        type=str,
        default="Training",
        choices=["Training", "PublicTest", "PrivateTest", "all"],
        help="Which CSV Usage rows to export (default: Training)",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing PNG/JPG under emotion subfolders of --out first",
    )
    args = p.parse_args()

    out_root = Path(args.out).resolve()

    if args.clean:
        for sub in out_root.glob("*"):
            if sub.is_dir():
                for img in sub.iterdir():
                    if img.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        img.unlink()

    if args.csv:
        csv_path = Path(args.csv).resolve()
        if not csv_path.is_file():
            raise SystemExit(f"CSV not found: {csv_path}")
        uf = None if args.usage == "all" else args.usage
        import_from_csv(csv_path, out_root, uf)
        return

    root = Path(args.csv_root).resolve()
    candidates = [
        root / "fer2013" / "fer2013.csv",
        root / "fer2013" / "icml_face_data.csv",
    ]
    csv_path = next((p for p in candidates if p.is_file()), None)
    if csv_path is None:
        raise SystemExit(
            "No CSV found. Either:\n"
            f"  1) Place fer2013.csv at: {candidates[0]}\n"
            "  2) Or pass --csv C:\\path\\to\\fer2013.csv\n"
            "Download from Kaggle (msambare/fer2013 or original challenge)."
        )
    uf = None if args.usage == "all" else args.usage
    import_from_csv(csv_path, out_root, uf)


if __name__ == "__main__":
    main()
