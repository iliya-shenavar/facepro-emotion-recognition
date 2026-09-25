from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


def _shuffle_split_indices(n: int, val_ratio: float, seed: int):
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    va = sorted(idx[-n_val:])
    tr = sorted(idx[:-n_val])
    return tr, va


def iter_imagefolder_samples(root: Path, subset_ratio: float, seed: int):
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths_labels = []
    for c in classes:
        for f in (root / c).iterdir():
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                paths_labels.append((f, class_to_idx[c]))
    rng = random.Random(seed)
    if subset_ratio < 1.0:
        k = max(1, int(len(paths_labels) * subset_ratio))
        paths_labels = rng.sample(paths_labels, k=k)
    return paths_labels, classes


def _clahe_equalize(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def hog_feature(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float64) / 255.0
    return hog(
        g,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def lbp_feature(gray: np.ndarray, p: int = 8, r: int = 1) -> np.ndarray:
    lbp = local_binary_pattern(gray, p, r, method="uniform")
    n_bins = p + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float64)


def combined_feature(gray: np.ndarray) -> np.ndarray:
    eq = _clahe_equalize(gray)
    hog_f = hog_feature(eq)
    lbp_f = lbp_feature(eq)
    return np.concatenate([hog_f, lbp_f], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/fer2013/train")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--subset_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca_dim", type=int, default=300)
    p.add_argument("--grid_search", action="store_true", default=True)
    p.add_argument("--no-grid_search", dest="grid_search", action="store_false")
    args = p.parse_args()

    root = Path(args.data_dir)
    if not root.is_dir():
        raise SystemExit(f"Missing data dir: {root}")

    samples, class_names = iter_imagefolder_samples(root, args.subset_ratio, args.seed)
    n = len(samples)
    tr_idx, va_idx = _shuffle_split_indices(n, args.val_ratio, args.seed)

    def xs_ys(indices):
        X, y = [], []
        for i in tqdm(indices, desc="features"):
            path, lab = samples[i]
            im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            im = cv2.resize(im, (args.img_size, args.img_size))
            feat = combined_feature(im)
            X.append(feat)
            y.append(lab)
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)

    X_train, y_train = xs_ys(tr_idx)
    X_val, y_val = xs_ys(va_idx)

    n_components = min(args.pca_dim, X_train.shape[0], X_train.shape[1])

    if args.grid_search:
        base_pipe = make_pipeline(
            StandardScaler(with_mean=True),
            PCA(n_components=n_components, random_state=args.seed),
            SVC(kernel="rbf", class_weight="balanced"),
        )
        param_grid = {
            "svc__C": [1, 5, 10, 20],
            "svc__gamma": ["scale", 0.01, 0.001],
        }
        search = GridSearchCV(base_pipe, param_grid, cv=3, n_jobs=-1, scoring="balanced_accuracy")
        search.fit(X_train, y_train)
        clf = search.best_estimator_
        print("Best params:", search.best_params_)
    else:
        clf = make_pipeline(
            StandardScaler(with_mean=True),
            PCA(n_components=n_components, random_state=args.seed),
            SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
        )
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=class_names, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
