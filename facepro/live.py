from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from facepro.models import build_model
from facepro.predict import pil_to_model_input


def open_camera(preferred_index: int, backend_name: str):
    backend_map = {
        "any": cv2.CAP_ANY,
        "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    }
    if backend_name != "auto":
        backends = [(backend_name, backend_map[backend_name])]
    else:
        backends = [("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW), ("any", cv2.CAP_ANY)]

    indices_to_try = [preferred_index] + [i for i in range(4) if i != preferred_index]

    for idx in indices_to_try:
        for name, backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    print(f"Opened camera index {idx} with backend '{name}'.")
                    return cap
            cap.release()
    return None


def load_checkpoint(path: str):
    kw = {"map_location": "cpu"}
    try:
        kw["weights_only"] = False
        return torch.load(path, **kw)
    except TypeError:
        return torch.load(path, map_location="cpu")


def expand_bbox(x: int, y: int, w: int, h: int, fw: int, fh: int, margin: float) -> tuple[int, int, int, int]:
    m = float(max(0.0, margin))
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    nw = int(w * (1.0 + m))
    nh = int(h * (1.0 + m))
    x0 = int(cx - 0.5 * nw)
    y0 = int(cy - 0.5 * nh)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(fw, x0 + nw)
    y1 = min(fh, y0 + nh)
    if x1 <= x0 or y1 <= y0:
        return x, y, w, h
    return x0, y0, x1 - x0, y1 - y0


def roi_to_pil(roi_bgr: np.ndarray, use_clahe: bool) -> Image.Image:
    if not use_clahe:
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    bgr2 = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def main():
    p = argparse.ArgumentParser(description="Live emotion from webcam")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture device index")
    p.add_argument(
        "--face_scale",
        type=float,
        default=1.1,
        help="Haar cascade scaleFactor (slightly bigger = faster, fewer detects)",
    )
    p.add_argument("--face_neighbors", type=int, default=5)
    p.add_argument("--min_face", type=int, default=80, help="Min face width in pixels")
    p.add_argument(
        "--every",
        type=int,
        default=2,
        help="Run the network every N frames (1 = each frame)",
    )
    p.add_argument(
        "--smooth",
        type=float,
        default=0.72,
        help="Temporal EMA on probabilities in [0,1); 0 = no smoothing",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.22,
        help="Expand face box by this fraction (0=off). Helps alignment vs training crops.",
    )
    p.add_argument(
        "--clahe",
        action="store_true",
        default=True,
        help="CLAHE contrast on face ROI (default on). Use --no-clahe to disable.",
    )
    p.add_argument("--no-clahe", dest="clahe", action="store_false")
    p.add_argument(
        "--vote",
        type=int,
        default=5,
        help="Majority vote over last N predictions (0 = off; reduces jitter).",
    )
    p.add_argument("--device", type=str, default="", help="cuda or cpu; default = auto")
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "msmf", "dshow", "any"],
        help="Windows camera backend to try (auto tries msmf, then dshow, then any).",
    )
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(str(ckpt_path))
    class_names = ckpt["class_names"]
    img_size = int(ckpt.get("img_size", 48))
    model = build_model(ckpt["model"], num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])

    device_s = args.device.strip().lower() or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)
    model = model.to(device)
    model.eval()

    haar_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(haar_path)
    if cascade.empty():
        raise SystemExit(f"Could not load Haar cascade from {haar_path}")

    cap = open_camera(args.camera, args.backend)
    if cap is None:
        raise SystemExit(
            "Could not open any camera (tried indices 0-3 with available backends).\n"
            "Check: 1) another app can access the camera (e.g. Windows Camera app), "
            "2) Settings > Privacy & security > Camera allows desktop apps, "
            "3) no other program is currently holding the camera."
        )

    probs_ema = None
    frame_i = -1
    vote_buf: deque[int] = deque(maxlen=max(1, int(args.vote))) if args.vote > 0 else deque(maxlen=1)

    win = "Facepro — emotion (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("Showing feed. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Capture ended.")
                break
            frame_i += 1
            fh, fw = frame.shape[0], frame.shape[1]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=args.face_scale,
                minNeighbors=args.face_neighbors,
                minSize=(args.min_face, args.min_face),
            )

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda fh_: fh_[2] * fh_[3])
                x, y, w, h = expand_bbox(x, y, w, h, fw, fh, args.margin)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
                roi_bgr = frame[y : y + h, x : x + w]
                pil = roi_to_pil(roi_bgr, args.clahe)

                if frame_i % max(1, args.every) == 0:
                    xb = pil_to_model_input(pil, img_size).to(device)
                    with torch.no_grad():
                        logits = model(xb)
                        pr = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
                    s = float(args.smooth)
                    if probs_ema is None or s <= 0:
                        probs_ema = pr
                    else:
                        probs_ema = (1.0 - s) * pr + s * probs_ema
                        probs_ema /= max(1e-8, float(np.sum(probs_ema)))

                if probs_ema is not None:
                    k_raw = int(np.argmax(probs_ema))
                    conf_raw = float(probs_ema[k_raw])
                    if args.vote > 0:
                        vote_buf.append(k_raw)
                        k = Counter(vote_buf).most_common(1)[0][0]
                        conf = float(probs_ema[k])
                    else:
                        k, conf = k_raw, conf_raw
                    label = f"{class_names[k]} ({100.0 * conf:.0f}%)"
                    cv2.putText(
                        frame,
                        label,
                        (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                probs_ema = None
                vote_buf.clear()
                cv2.putText(
                    frame,
                    "No face",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
