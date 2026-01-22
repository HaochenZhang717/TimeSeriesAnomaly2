import argparse
import torch
import json
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
    parser.add_argument("--indices_paths_train", type=json.loads, required=True)
    parser.add_argument("--indices_paths_test", type=json.loads, required=True)
    parser.add_argument("--max_infill_length", type=int, required=True)


    """save and load parameters"""
    parser.add_argument("--out_dir", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)

    return parser.parse_args()



def extract_data_and_labels(real_set: ImputationECGDataset, device, one_channel):
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
    if one_channel:
        real_data = real_data[:, :, :1]
    real_labels = torch.cat(real_labels, dim=0).to(device=device)

    return real_data, real_labels


def save_real_data(
        raw_data_paths,
        indices_paths_test,
        seq_len,
        one_channel,
        max_infill_length,
        save_path
):
    device = 'cpu'
    real_set = ImputationECGDataset(
        raw_data_paths=raw_data_paths,
        indices_paths=indices_paths_test,
        seq_len=seq_len,
        one_channel=one_channel,
        max_infill_length=max_infill_length,
    )
    real_data, real_labels = extract_data_and_labels(real_set, device, one_channel)
    print("-" * 100)
    print("real data shape:", real_data.shape)
    print("real labels shape:", real_labels.shape)
    print("-" * 100)
    to_save = {
        "all_samples": real_data,
        "all_labels": real_labels
    }
    torch.save(to_save, save_path)

def main():
    # save_real_data(
    #     raw_data_paths=["./dataset_utils/ECG_datasets/raw_data/106.npz"],
    #     indices_paths_test=["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_test.jsonl"],
    #     seq_len=1000,
    #     one_channel=0,
    #     max_infill_length=800,
    #     save_path="./dataset_utils/real_anomaly_data_test/mitdb.pth"
    # )

    save_real_data(
        raw_data_paths=["./dataset_utils/ECG_datasets/raw_data/106.npz"],
        indices_paths_test=["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_train.jsonl"],
        seq_len=1000,
        one_channel=0,
        max_infill_length=800,
        save_path="./dataset_utils/real_anomaly_data_test/mitdb_train.pth"
    )


    # save_real_data(
    #     raw_data_paths=["./dataset_utils/ECG_datasets/raw_data_qtdb/sel233.npz"],
    #     indices_paths_test=["./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_test.jsonl"],
    #     seq_len=600,
    #     one_channel=0,
    #     max_infill_length=450,
    #     save_path="./dataset_utils/real_anomaly_data_test/qtdb.pth"
    # )
    #
    # save_real_data(
    #     raw_data_paths=["./dataset_utils/ECG_datasets/raw_data_svdb/859.npz"],
    #     indices_paths_test=["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/V_test.jsonl"],
    #     seq_len=800,
    #     one_channel=0,
    #     max_infill_length=360,
    #     save_path="./dataset_utils/real_anomaly_data_test/svdb.pth"
    # )
    #
    # save_real_data(
    #     raw_data_paths=["./dataset_utils/ECG_datasets/raw_data_traffic/metro_traffic_data.npz"],
    #     indices_paths_test=["./dataset_utils/ECG_datasets/indices_traffic/slide_windows_metro_traffic_datanpz/V_test.jsonl"],
    #     seq_len=72,
    #     one_channel=1,
    #     max_infill_length=24,
    #     save_path="./dataset_utils/real_anomaly_data_test/traffic.pth"
    # )
    #
    #
    #
    # save_real_data(
    #     raw_data_paths=["./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_2.npz","./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_3.npz","./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_4.npz","./dataset_utils/ECG_datasets/raw_data_PV/2015_pv_sub_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2015_pv_sub_1.npz","./dataset_utils/ECG_datasets/raw_data_PV/2021_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2022_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2023_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2024_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2025_pv_live_0.npz"],
    #     indices_paths_test=["./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/V_test.jsonl"],
    #     seq_len=200,
    #     one_channel=1,
    #     max_infill_length=144,
    #     save_path="./dataset_utils/real_anomaly_data_test/pv.pth"
    # )

if __name__ == "__main__":
    main()