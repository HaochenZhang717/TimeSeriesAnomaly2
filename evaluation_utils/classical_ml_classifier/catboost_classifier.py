import os
import json
import numpy as np
import torch

from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_fscore_support



def extract_cb_features(x, window):
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

    return np.stack(feats)



def build_cb_dataset(data, labels, window):
    X_all, y_all = [], []

    for x, y in zip(data, labels):
        x_np = x if isinstance(x, np.ndarray) else x.cpu().numpy()
        y_np = y if isinstance(y, np.ndarray) else y.cpu().numpy()

        feats = extract_cb_features(x_np, window)
        X_all.append(feats)
        y_all.append(y_np)

    return np.concatenate(X_all), np.concatenate(y_all)



def run_catboost_evaluate(args, real_data, real_labels, gen_data, gen_labels):
    output_record = {
        "args": vars(args),
    }

    precisions = []
    recalls = []
    f1s = []
    normal_accuracies = []
    anomaly_accuracies = []

    for _ in range(1):
        # ---- sample generated data ----
        random_indices = torch.randperm(len(gen_data))[:50000]
        sampled_gen_data = gen_data[random_indices]
        sampled_gen_labels = gen_labels[random_indices]

        print("real_data.shape:", real_data.shape)
        print("real_labels.shape:", real_labels.shape)
        print("gen_data.shape:", gen_data.shape)
        print("gen_labels.shape:", gen_labels.shape)

        # ---- build training data ----
        X_real, y_real = build_cb_dataset(
            real_data, real_labels, window=args.feat_window_size
        )
        X_gen, y_gen = build_cb_dataset(
            sampled_gen_data, sampled_gen_labels, window=args.feat_window_size
        )

        X_train = np.concatenate([X_real, X_gen])
        y_train = np.concatenate([y_real, y_gen])

        # ---- CatBoost ----
        model = CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="F1",
            auto_class_weights="Balanced",
            random_seed=0,
            verbose=False,   # 🔑 不刷屏
        )

        model.fit(X_train, y_train)

        # ---- test on real data only ----
        X_test, y_test = build_cb_dataset(
            real_data, real_labels, window=args.feat_window_size
        )
        y_pred = model.predict(X_test)
        y_pred = y_pred.astype(int)

        # ---- metrics ----
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )

        normal_acc = (y_pred[y_test == 0] == 0).mean()
        anomaly_acc = (y_pred[y_test == 1] == 1).mean()

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        normal_accuracies.append(normal_acc)
        anomaly_accuracies.append(anomaly_acc)

    # ---- aggregate ----
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)
    mean_normal_accuracy = np.mean(normal_accuracies)
    mean_anomaly_accuracy = np.mean(anomaly_accuracies)

    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1s)
    std_normal_accuracy = np.std(normal_accuracies)
    std_anomaly_accuracy = np.std(anomaly_accuracies)

    print(f"precision: {mean_precision}+-{std_precision}")
    print(f"recall: {mean_recall}+-{std_recall}")
    print(f"f1: {mean_f1}+-{std_f1}")
    print(f"normal_accuracy: {mean_normal_accuracy}+-{std_normal_accuracy}")
    print(f"anomaly_accuracy: {mean_anomaly_accuracy}+-{std_anomaly_accuracy}")

    result = {
        "precision_mean": float(mean_precision),
        "precision_std": float(std_precision),
        "recall_mean": float(mean_recall),
        "recall_std": float(std_recall),
        "f1_mean": float(mean_f1),
        "f1_std": float(std_f1),
        "normal_accuracy_mean": float(mean_normal_accuracy),
        "normal_accuracy_std": float(std_normal_accuracy),
        "anomaly_accuracy_mean": float(mean_anomaly_accuracy),
        "anomaly_accuracy_std": float(std_anomaly_accuracy),
    }

    output_record.update({"result_CatBoost": result})

    save_path = os.path.join(args.out_dir, "catboost_evaluation_results.jsonl")
    with open(save_path, "a") as f:
        json.dump(output_record, f, indent=2)
        f.write("\n")