import argparse
import torch
import sys

# # 1. 强行从 sys.modules 中彻底移除所有与 apex 相关的记录
# # 即使之前 import 失败留下了残余，也要清理干净
# for key in list(sys.modules.keys()):
#     if 'apex' in key:
#         del sys.modules[key]
#
# # 2. 伪造一个“绝对不可用”的导入陷阱
# # 我们创建一个名为 apex 的类，但它不具备任何属性，
# # 关键是我们要让它在 transformers 检测时抛出真正的 ImportError
# class InvisibleModule:
#     pass
#
# # 3. 告诉 Python，apex 已经加载过了，但它是个空壳
# # 这样 transformers 的 import apex 语句会成功，
# # 但它后续检查 hasattr(apex, 'normalization') 时会返回 False
# sys.modules["apex"] = InvisibleModule()
#
# # 4. 最关键的一步：防止 importlib.util.find_spec("apex") 崩溃
# # 我们手动设置 sys.modules["apex"].__path__ 为空，让它看起来不像一个包
# sys.modules["apex"].__path__ = []



import json
import os
import numpy as np
from torch.utils.data import Subset
from evaluation_utils import calculate_LSTM, calculate_GRU, calculate_robustTAD, calculate_TCN, calculate_Transformer
from evaluation_utils import run_GPT4TS_evaluate, run_moment_evaluate
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


def run_lstm_evaluate(args, real_data, real_labels, gen_data, gen_labels, device):
    output_record = {
        "args": vars(args),
    }

    precisions = []
    recalls = []
    f1s = []
    normal_accuracies = []
    anomaly_accuracies = []
    for _ in range(5):
        random_indices = torch.randperm(len(gen_data))[:50000]
        sampled_gen_data = gen_data[random_indices]
        sampled_gen_labels = gen_labels[random_indices]

        print("real_data.shape:", real_data.shape)
        print("real_labels.shape:", real_labels.shape)
        print("gen_data.shape:", gen_data.shape)
        print("gen_labels.shape:", gen_labels.shape)

        normal_accuracy, anomaly_accuracy, precision, recall, f1 = calculate_LSTM(
            anomaly_weight=1.0,
            feature_size=args.feature_size,
            ori_data=real_data,
            ori_labels=real_labels,
            gen_data=sampled_gen_data,
            gen_labels=sampled_gen_labels,
            device=device,
            lr=1e-5,
            max_epochs=2000,
            batch_size=64,
            patience=20)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        normal_accuracies.append(normal_accuracy)
        anomaly_accuracies.append(anomaly_accuracy)

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
    output_record.update({"result_LSTM": result})

    save_path = os.path.join(args.out_dir, f"lstm_evaluation_results.jsonl")


    with open(save_path, "a") as f:
        json.dump(output_record, f, indent=2)
        f.write("\n")


def run_gru_evaluate(args, real_data, real_labels, gen_data, gen_labels, device):
    output_record = {
        "args": vars(args),
    }

    precisions = []
    recalls = []
    f1s = []
    normal_accuracies = []
    anomaly_accuracies = []
    for _ in range(5):
        random_indices = torch.randperm(len(gen_data))[:50000]
        sampled_gen_data = gen_data[random_indices]
        sampled_gen_labels = gen_labels[random_indices]

        print("real_data.shape:", real_data.shape)
        print("real_labels.shape:", real_labels.shape)
        print("gen_data.shape:", gen_data.shape)
        print("gen_labels.shape:", gen_labels.shape)

        normal_accuracy, anomaly_accuracy, precision, recall, f1 = calculate_GRU(
            anomaly_weight=1.0,
            feature_size=args.feature_size,
            ori_data=real_data,
            ori_labels=real_labels,
            gen_data=sampled_gen_data,
            gen_labels=sampled_gen_labels,
            device=device,
            lr=1e-5,
            max_epochs=2000,
            batch_size=64,
            patience=20)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        normal_accuracies.append(normal_accuracy)
        anomaly_accuracies.append(anomaly_accuracy)

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
    output_record.update({"result_GRU": result})

    save_path = os.path.join(args.out_dir, f"gru_evaluation_results.jsonl")

    # with open(save_path, "a") as f:
    #     f.write(json.dumps(output_record) + "\n")

    with open(save_path, "a") as f:
        json.dump(output_record, f, indent=2)
        f.write("\n")


def run_robustTAD_evaluate(args, real_data, real_labels, gen_data, gen_labels, device):
    output_record = {
        "args": vars(args),
    }

    precisions = []
    recalls = []
    f1s = []
    normal_accuracies = []
    anomaly_accuracies = []
    for _ in range(5):
        random_indices = torch.randperm(len(gen_data))[:10000]
        sampled_gen_data = gen_data[random_indices]
        sampled_gen_labels = gen_labels[random_indices]

        print("real_data.shape:", real_data.shape)
        print("real_labels.shape:", real_labels.shape)
        print("gen_data.shape:", gen_data.shape)
        print("gen_labels.shape:", gen_labels.shape)

        normal_accuracy, anomaly_accuracy, precision, recall, f1 = calculate_robustTAD(
            anomaly_weight=1.0,
            feature_size=args.feature_size,
            ori_data=real_data,
            ori_labels=real_labels,
            gen_data=sampled_gen_data,
            gen_labels=sampled_gen_labels,
            device=device,
            lr=1e-4,
            min_lr=5e-6,
            max_epochs=2000,
            batch_size=64,
            patience=100)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        normal_accuracies.append(normal_accuracy)
        anomaly_accuracies.append(anomaly_accuracy)

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
    output_record.update({"result_robustTAD": result})

    save_path = os.path.join(args.out_dir, f"robusttad_evaluation_results.jsonl")

    # with open(save_path, "a") as f:
    #     f.write(json.dumps(output_record) + "\n")

    with open(save_path, "a") as f:
        json.dump(output_record, f, indent=2)
        f.write("\n")


def run_TCN_evaluate(args, real_data, real_labels, gen_data, gen_labels, device):
    output_record = {
        "args": vars(args),
    }

    precisions = []
    recalls = []
    f1s = []
    normal_accuracies = []
    anomaly_accuracies = []
    for _ in range(5):
        random_indices = torch.randperm(len(gen_data))[:10000]
        sampled_gen_data = gen_data[random_indices]
        sampled_gen_labels = gen_labels[random_indices]

        print("real_data.shape:", real_data.shape)
        print("real_labels.shape:", real_labels.shape)
        print("gen_data.shape:", gen_data.shape)
        print("gen_labels.shape:", gen_labels.shape)

        normal_accuracy, anomaly_accuracy, precision, recall, f1 = calculate_TCN(
            anomaly_weight=1.0,
            feature_size=args.feature_size,
            ori_data=real_data,
            ori_labels=real_labels,
            gen_data=sampled_gen_data,
            gen_labels=sampled_gen_labels,
            device=device,
            lr=1e-4,
            min_lr=5e-6,
            lr_decay_patience=10,
            max_epochs=2000,
            batch_size=64,
            patience=100)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        normal_accuracies.append(normal_accuracy)
        anomaly_accuracies.append(anomaly_accuracy)

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
    output_record.update({"result_TCN": result})

    save_path = os.path.join(args.out_dir, f"TCN_evaluation_results.jsonl")

    # with open(save_path, "a") as f:
    #     f.write(json.dumps(output_record) + "\n")

    with open(save_path, "a") as f:
        json.dump(output_record, f, indent=2)
        f.write("\n")


def run_transformer_evaluate(args, real_data, real_labels, gen_data, gen_labels, device):
    output_record = {
        "args": vars(args),
    }

    precisions = []
    recalls = []
    f1s = []
    normal_accuracies = []
    anomaly_accuracies = []
    for _ in range(1):
        random_indices = torch.randperm(len(gen_data))[:50000]
        sampled_gen_data = gen_data[random_indices]
        sampled_gen_labels = gen_labels[random_indices]

        print("real_data.shape:", real_data.shape)
        print("real_labels.shape:", real_labels.shape)
        print("gen_data.shape:", gen_data.shape)
        print("gen_labels.shape:", gen_labels.shape)

        normal_accuracy, anomaly_accuracy, precision, recall, f1 = calculate_Transformer(
            anomaly_weight=1.0,
            feature_size=args.feature_size,
            ori_data=real_data,
            ori_labels=real_labels,
            gen_data=sampled_gen_data,
            gen_labels=sampled_gen_labels,
            device=device,
            lr=1e-5,
            max_epochs=2000,
            batch_size=64,
            patience=20)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        normal_accuracies.append(normal_accuracy)
        anomaly_accuracies.append(anomaly_accuracy)

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
    output_record.update({"result_transformer": result})

    save_path = os.path.join(args.out_dir, f"transformer_evaluation_results.jsonl")

    # with open(save_path, "a") as f:
    #     f.write(json.dumps(output_record) + "\n")

    with open(save_path, "a") as f:
        json.dump(output_record, f, indent=2)
        f.write("\n")




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


    # run_robustTAD_evaluate(args, real_data, real_labels, gen_data, gen_labels, device)
    run_TCN_evaluate(args, real_data, real_labels, gen_data, gen_labels, device)
    # run_GPT4TS_evaluate(args, real_data, real_labels, gen_data, gen_labels, device)
    # run_moment_evaluate(
    #     real_data, real_labels, gen_data, gen_labels,
    #     model_name="large", one_channel=args.one_channel,
    #     output_path=args.out_dir
    # )

    # run_rf_evaluate(args, real_data, real_labels, gen_data, gen_labels)
    # run_catboost_evaluate(args, real_data, real_labels, gen_data, gen_labels)

    print("all done")

if __name__ == "__main__":
    main()