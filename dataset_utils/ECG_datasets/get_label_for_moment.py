import json
import numpy as np
import torch

def mitdb():
    path = "./indices/slide_windows_106npz/train/V_test.jsonl"
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    all_samples = []
    all_labels = []
    raw_data = np.load("./raw_data/106.npz")
    raw_signal = raw_data["signal"]
    anomaly_label = raw_data["anomaly_label"]

    for record in records:
        start = record["ts_start"]
        end = record["ts_end"]
        all_labels.append(anomaly_label[start:end])
        all_samples.append(raw_signal[start:end])

    all_samples = torch.from_numpy(np.stack(all_samples, axis=0))
    all_labels = torch.from_numpy(np.stack(all_labels, axis=0))

    to_save = {
        "all_samples": all_samples,
        "all_labels": all_labels,
    }
    torch.save(to_save, "test_data/mitdb106_test_data.pt")


def qtdb():
    path = "./indices_qtdb/slide_windows_sel233npz/V_test.jsonl"
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    all_samples = []
    all_labels = []
    raw_data = np.load("./raw_data_qtdb/sel233.npz")
    raw_signal = raw_data["signal"]
    anomaly_label = raw_data["anomaly_label"]

    for record in records:
        start = record["ts_start"]
        end = record["ts_end"]
        all_labels.append(anomaly_label[start:end])
        all_samples.append(raw_signal[start:end])

    all_samples = torch.from_numpy(np.stack(all_samples, axis=0))
    all_labels = torch.from_numpy(np.stack(all_labels, axis=0))

    to_save = {
        "all_samples": all_samples,
        "all_labels": all_labels,
    }
    torch.save(to_save, "test_data/qtdb233_test_data.pt")

if __name__ == "__main__":
    qtdb()