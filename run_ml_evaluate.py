import argparse
import torch
import json
import os
import numpy as np
from torch.utils.data import Subset
from evaluation_utils import run_rf_evaluate, run_catboost_evaluate
from dataset_utils import ImputationECGDataset


def get_args():
    parser = argparse.ArgumentParser(description="parameters for flow-ts pretraining")

    """time series general parameters"""
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--feature_size", type=int, required=True)
    parser.add_argument("--one_channel", type=int, required=True)
    parser.add_argument("--feat_window_size", type=int, required=True)


    """data parameters"""
    parser.add_argument("--raw_data_paths", type=json.loads, required=True)
    parser.add_argument("--indices_paths_test", type=json.loads, required=True)
    parser.add_argument("--max_infill_length", type=int, required=True)


    """save and load parameters"""
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--generated_path", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)

    return parser.parse_args()



def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu_id}")
    os.makedirs(args.out_dir, exist_ok=True)

    real_set = ImputationECGDataset(
        raw_data_paths=args.raw_data_paths,
        indices_paths=args.indices_paths_test,
        seq_len=args.seq_len,
        one_channel=args.one_channel,
        max_infill_length=args.max_infill_length,
    )

    real_data = []
    real_labels = []
    for which_list, which_index in real_set.global_index:
        ts_start = real_set.index_lines_list[which_list][which_index]["ts_start"]
        ts_end = real_set.index_lines_list[which_list][which_index]["ts_end"]
        anomaly_start = real_set.index_lines_list[which_list][which_index]["anomaly_start"]
        anomaly_end = real_set.index_lines_list[which_list][which_index]["anomaly_end"]

        relative_anomaly_start = anomaly_start - ts_start
        relative_anomaly_end = anomaly_end - ts_start
        real_datum = torch.from_numpy(real_set.normed_signal_list[which_list][ts_start:ts_end])
        real_label = torch.zeros(len(real_datum)).to(device=device)
        real_label[relative_anomaly_start:relative_anomaly_end] = 1

        real_data.append(real_datum.unsqueeze(0))
        real_labels.append(real_label.unsqueeze(0))

    real_data = torch.cat(real_data, dim=0).to(device=device)
    if args.one_channel:
        real_data = real_data[:,:,:1]
    real_labels = torch.cat(real_labels, dim=0).to(device=device)


    all_anomalies = torch.load(args.generated_path, map_location=device)


    gen_data = all_anomalies['all_samples']
    gen_labels = all_anomalies['all_labels']

    # ---- Step 1: 找出含 NaN 的样本 ----
    nan_mask = torch.isnan(gen_data).any(dim=(1, 2))  # True 表示该样本含 NaN

    print("Samples containing NaN:", nan_mask.sum().item(), "/", gen_data.size(0))
    # ---- Step 2: 删除这些样本 ----
    gen_data = gen_data[~nan_mask]
    gen_labels = gen_labels[~nan_mask]



    run_rf_evaluate(args, real_data, real_labels, gen_data, gen_labels)
    run_catboost_evaluate(args, real_data, real_labels, gen_data, gen_labels)



if __name__ == "__main__":
    main()