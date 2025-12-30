import os
import json
import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support


def extract_rf_features(x, window=20):
    """
    x: [T, C] or [T]
    return: [T, D]
    """
    if x.ndim == 1:
        x = x[:, None]

    T, C = x.shape
    pad = window
    x_pad = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")

    feats = []
    for t in range(T):
        seg = x_pad[t : t + 2 * window + 1]   # [2w+1, C]

        f_raw = seg.reshape(-1)
        f_mean = seg.mean(axis=0)
        f_std = seg.std(axis=0)
        f_diff = np.diff(seg, axis=0).mean(axis=0)

        feats.append(np.concatenate([f_raw, f_mean, f_std, f_diff]))

    return np.stack(feats)   # [T, D]



def build_rf_dataset(data, labels, window=20):
    """
    data:   [N, T, C]
    labels: [N, T]
    return: X [num_points, D], y [num_points]
    """
    X_all, y_all = [], []

    for x, y in zip(data, labels):
        x_np = x if isinstance(x, np.ndarray) else x.cpu().numpy()
        y_np = y if isinstance(y, np.ndarray) else y.cpu().numpy()

        feats = extract_rf_features(x_np, window)
        X_all.append(feats)
        y_all.append(y_np)

    return np.concatenate(X_all), np.concatenate(y_all)


