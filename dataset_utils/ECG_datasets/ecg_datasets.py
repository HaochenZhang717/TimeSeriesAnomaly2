import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, IterableDataset, DataLoader
import json
import matplotlib.pyplot as plt
import random
import math


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data



class ECGDataset(Dataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            seq_len,
            max_anomaly_length,
            min_anomaly_length,
            one_channel,
            # limited_data_size,
    ):
        super(ECGDataset, self).__init__()
        self.seq_len = seq_len
        # self.max_anomaly_ratio = max_anomaly_ratio
        self.max_anomaly_length = max_anomaly_length
        self.min_anomaly_length = min_anomaly_length
        self.one_channel = one_channel
        # self.limited_data_size = limited_data_size
        self.slide_windows = []
        self.anomaly_labels = []

        indices_paths = [indices_paths]
        raw_data_paths = [raw_data_paths]

        for indices_path, raw_data_path in zip(indices_paths, raw_data_paths):

            raw_data = np.load(raw_data_path)
            raw_signal = raw_data["signal"]
            anomaly_label = raw_data["anomaly_label"]
            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines= load_jsonl(indices_path)

            for index_line in index_lines:
                start_index = index_line["start"]
                end_index = index_line["end"]
                self.slide_windows.append(normed_signal[start_index:end_index])
                self.anomaly_labels.append(anomaly_label[start_index:end_index])
                # if len(self.slide_windows) >= self.limited_data_size:
                #     break

    def __getitem__(self, index):
        signal = self.slide_windows[index]
        anomaly_label = self.anomaly_labels[index]
        assert len(np.unique(anomaly_label)) <= 2
        anomaly_label = (anomaly_label > 0).astype(np.int8)

        random_anomaly_length = np.random.randint(self.min_anomaly_length, self.max_anomaly_length)
        anomaly_start = np.random.randint(0, self.max_anomaly_length - random_anomaly_length)
        anomaly_end = anomaly_start + random_anomaly_length

        random_anomaly_label = np.zeros_like(anomaly_label)
        random_anomaly_label[anomaly_start:anomaly_end] = 1
        signal_random_occluded = signal * (1 - random_anomaly_label[:, None])
        original_occluded_signal = signal * (1 - anomaly_label[:, None])
        if not self.one_channel:
            sample = {
                "orig_signal": signal,
                "anomaly_label": anomaly_label,
                "original_occluded_signal": original_occluded_signal,
                "random_anomaly_label": random_anomaly_label,
                "signal_random_occluded": signal_random_occluded,
            }
        else:
            sample = {
                "orig_signal": signal[:, :1],
                "anomaly_label": anomaly_label,
                "original_occluded_signal": original_occluded_signal[:, :1],
                "random_anomaly_label": random_anomaly_label,
                "signal_random_occluded": signal_random_occluded[:, :1],
            }
        return sample

    def __len__(self):
        return len(self.slide_windows)



class IterableECGDataset(IterableDataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            seq_len,
            max_anomaly_length,
            min_anomaly_length,
            one_channel: bool,
            limited_data_size: int,
    ):
        super(IterableECGDataset, self).__init__()
        self.seq_len = seq_len
        self.one_channel = one_channel
        self.limited_data_size = limited_data_size
        # self.max_anomaly_ratio = max_anomaly_ratio
        self.max_anomaly_length = max_anomaly_length
        self.min_anomaly_length = min_anomaly_length
        self.slide_windows = []
        self.anomaly_labels = []

        indices_paths = [indices_paths]
        raw_data_paths = [raw_data_paths]

        for indices_path, raw_data_path in zip(indices_paths, raw_data_paths):

            raw_data = np.load(raw_data_path)
            raw_signal = raw_data["signal"]
            anomaly_label = raw_data["anomaly_label"]
            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines= load_jsonl(indices_path)

            for index_line in index_lines:
                start_index = index_line["start"]
                end_index = index_line["end"]
                self.slide_windows.append(normed_signal[start_index:end_index])
                self.anomaly_labels.append(anomaly_label[start_index:end_index])
                if len(self.slide_windows) >= self.limited_data_size:
                    break

    def __iter__(self):
        while True:
            index = np.random.randint(len(self.slide_windows))
            signal = self.slide_windows[index]
            anomaly_label = self.anomaly_labels[index]
            assert len(np.unique(anomaly_label)) <= 2
            anomaly_label = (anomaly_label > 0).astype(np.int8)

            random_anomaly_length = np.random.randint(self.min_anomaly_length, self.max_anomaly_length)
            anomaly_start = np.random.randint(0, self.max_anomaly_length - random_anomaly_length)
            anomaly_end = anomaly_start + random_anomaly_length

            random_anomaly_label = np.zeros_like(anomaly_label)
            random_anomaly_label[anomaly_start:anomaly_end] = 1
            signal_random_occluded = signal * (1 - random_anomaly_label[:, None])
            original_occluded_signal = signal * (1 - anomaly_label[:, None])
            if not self.one_channel:
                sample = {
                    "orig_signal": signal,
                    "anomaly_label": anomaly_label,
                    "original_occluded_signal": original_occluded_signal,
                    "random_anomaly_label": random_anomaly_label,
                    "signal_random_occluded": signal_random_occluded,
                }
            else:
                sample = {
                    "orig_signal": signal[:, :1],
                    "anomaly_label": anomaly_label,
                    "original_occluded_signal": original_occluded_signal[:, :1],
                    "random_anomaly_label": random_anomaly_label,
                    "signal_random_occluded": signal_random_occluded[:, :1],
                }

            yield sample



class NoContextECGDataset(Dataset):
    def __init__(
            self,
            raw_data_path,
            indices_path,
            seq_len,
            one_channel,
    ):
        super(NoContextECGDataset, self).__init__()
        self.seq_len = seq_len
        self.one_channel = one_channel

        raw_data = np.load(raw_data_path)
        raw_signal = raw_data["signal"]
        self.anomaly_label = raw_data["anomaly_label"]
        scaler = MinMaxScaler()
        self.normed_signal = scaler.fit_transform(raw_signal)
        self.index_lines= load_jsonl(indices_path)


    def __getitem__(self, index):

        start = self.index_lines[index]['start']
        end = self.index_lines[index]['end']

        if 'prototype_id' in self.index_lines[index].keys():
            prototype_id = self.index_lines[index]['prototype_id']
        else:
            prototype_id = -100

        if self.one_channel:
            signal = torch.from_numpy(self.normed_signal[start:end, :1])
        else:
            signal = torch.from_numpy(self.normed_signal[start:end])
        anomaly_label = self.anomaly_label[start:end]
        assert min(anomaly_label) == max(anomaly_label)

        return {'signal': signal, 'prototype_id': prototype_id}

    def __len__(self):
        return len(self.index_lines)



class ImputationECGDataset(Dataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            seq_len,
            one_channel,
            # use_prototype,
            max_infill_length,
    ):
        super(ImputationECGDataset, self).__init__()
        self.seq_len = seq_len
        self.one_channel = one_channel
        # self.use_prototype = use_prototype
        self.max_infill_length = max_infill_length
        self.raw_data_paths = raw_data_paths
        self.indices_paths = indices_paths

        self.normed_signal_list = []
        self.index_lines_list = []
        self.anomaly_label_list = []

        for raw_data_path, indices_path in zip(self.raw_data_paths, self.indices_paths):
            raw_data = np.load(raw_data_path)
            raw_signal = raw_data["signal"]
            anomaly_label = raw_data["anomaly_label"]

            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines = load_jsonl(indices_path)

            self.normed_signal_list.append(normed_signal)
            self.anomaly_label_list.append(anomaly_label)
            self.index_lines_list.append(index_lines)


        self.global_index = []
        for region_id, index_lines in enumerate(self.index_lines_list):
            for i in range(len(index_lines)):
                self.global_index.append((region_id, i))

        # raw_data = np.load(raw_data_path)
        # raw_signal = raw_data["signal"]
        # self.anomaly_label = raw_data["anomaly_label"]
        # scaler = MinMaxScaler()
        # self.normed_signal = scaler.fit_transform(raw_signal)
        # self.index_lines= load_jsonl(indices_path)


    def __getitem__(self, index):
        which_list, which_index = self.global_index[index]

        ts_start = self.index_lines_list[which_list][which_index]["ts_start"]
        ts_end = self.index_lines_list[which_list][which_index]["ts_end"]
        anomaly_start = self.index_lines_list[which_list][which_index]["anomaly_start"]
        anomaly_end = self.index_lines_list[which_list][which_index]["anomaly_end"]

        relative_anomaly_start = anomaly_start - ts_start
        relative_anomaly_end = anomaly_end - ts_start

        if self.one_channel:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end, :1])
        else:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end])
        # normalize each slide window
        # scaler = MinMaxScaler()
        # signal = scaler.fit_transform(signal)

        anomaly_label = torch.from_numpy(self.anomaly_label_list[which_list][ts_start:ts_end])
        T = anomaly_label.shape[0]
        # ===== attention mask =====
        # normal + target anomaly are visible
        context_mask = torch.zeros(T, dtype=torch.long)
        context_mask[anomaly_label == 0] = 1
        context_mask[relative_anomaly_start:relative_anomaly_end] = 1

        # plt.plot(anomaly_label, label="anomaly_label")
        # plt.plot(context_mask, label="context_mask")
        # plt.show()
        # ===== noise mask =====
        noise_mask = torch.zeros(T, dtype=torch.long)
        noise_mask[relative_anomaly_start:relative_anomaly_end] = 1

        # ===== missing signals =====
        missing_signals = torch.zeros(self.max_infill_length, signal.shape[-1])
        infill_length = anomaly_end - anomaly_start
        missing_signals[:infill_length] = signal[relative_anomaly_start:relative_anomaly_end]

        missing_signals_mask = torch.zeros(self.max_infill_length)
        missing_signals_mask[:infill_length] = 1

        return {
            'signals': signal,
            'missing_signals': missing_signals,
            'missing_signals_mask': missing_signals_mask,
            # 'prototypes': prototype_id,
            'attn_mask': context_mask,
            'noise_mask': noise_mask,
        }


    def __len__(self):
        return len(self.global_index)


class PredictionECGDataset(Dataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            seq_len,
            one_channel,
            max_infill_length,
            pre_context_length,
    ):
        super(PredictionECGDataset, self).__init__()
        self.seq_len = seq_len
        self.one_channel = one_channel
        self.pre_context_length = pre_context_length
        self.max_infill_length = max_infill_length
        self.raw_data_paths = raw_data_paths
        self.indices_paths = indices_paths

        self.normed_signal_list = []
        self.index_lines_list = []
        self.anomaly_label_list = []

        for raw_data_path, indices_path in zip(self.raw_data_paths, self.indices_paths):
            raw_data = np.load(raw_data_path)
            raw_signal = raw_data["signal"]
            anomaly_label = raw_data["anomaly_label"]

            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines = load_jsonl(indices_path)

            self.normed_signal_list.append(normed_signal)
            self.anomaly_label_list.append(anomaly_label)
            self.index_lines_list.append(index_lines)


        self.global_index = []
        for region_id, index_lines in enumerate(self.index_lines_list):
            for i in range(len(index_lines)):
                self.global_index.append((region_id, i))

    def __getitem__(self, index):
        which_list, which_index = self.global_index[index]

        # ts_start = self.index_lines_list[which_list][which_index]["ts_start"]
        # ts_end = self.index_lines_list[which_list][which_index]["ts_end"]
        anomaly_start = self.index_lines_list[which_list][which_index]["start"]
        anomaly_end = self.index_lines_list[which_list][which_index]["end"]
        context_start = anomaly_start - self.pre_context_length

        relative_anomaly_start = anomaly_start - context_start
        relative_anomaly_end = anomaly_end - context_start



        if self.one_channel:
            real_signal = torch.from_numpy(self.normed_signal_list[which_list][context_start:anomaly_end, :1])
        else:
            real_signal = torch.from_numpy(self.normed_signal_list[which_list][context_start:anomaly_end])

        # ===== signal =====
        signal = torch.zeros(self.seq_len, real_signal.shape[-1])
        signal[:anomaly_end - context_start] = real_signal
        # ===== anomaly label =====
        anomaly_label = torch.from_numpy(self.anomaly_label_list[which_list][context_start:anomaly_end])
        T = anomaly_label.shape[0]
        # ===== attention mask =====
        context_mask = torch.zeros(self.seq_len, dtype=torch.long)
        context_mask[torch.where(anomaly_label == 0)] = 1
        context_mask[relative_anomaly_start:relative_anomaly_end] = 1
        # ===== noise mask =====
        noise_mask = torch.zeros(self.seq_len, dtype=torch.long)
        noise_mask[relative_anomaly_start:relative_anomaly_end] = 1


        plt.plot(signal[:, 1], label='signal')
        # plt.plot(anomaly_label, label='anomaly_label')
        plt.plot(context_mask, label='context_mask')
        plt.plot(noise_mask, label='noise_mask')
        plt.legend()
        plt.show()


        # ===== missing signals =====
        missing_signals = torch.zeros(self.max_infill_length, signal.shape[-1])
        infill_length = anomaly_end - anomaly_start
        missing_signals[:infill_length] = signal[relative_anomaly_start:relative_anomaly_end]

        missing_signals_mask = torch.zeros(self.max_infill_length)
        missing_signals_mask[:infill_length] = 1

        return {
            'signals': signal,
            'missing_signals': missing_signals,
            'missing_signals_mask': missing_signals_mask,
            'attn_mask': context_mask,
            'noise_mask': noise_mask,
        }


    def __len__(self):
        return len(self.global_index)


class ImputationNormalECGDataset(Dataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            seq_len,
            one_channel,
            min_infill_length,
            max_infill_length,
    ):
        super(ImputationNormalECGDataset, self).__init__()
        self.seq_len = seq_len
        self.one_channel = one_channel
        self.raw_data_paths = raw_data_paths
        self.indices_paths = indices_paths
        self.min_infill_length = min_infill_length
        self.max_infill_length = max_infill_length

        self.normed_signal_list = []
        self.index_lines_list = []
        self.anomaly_label_list = []

        for raw_data_path, indices_path in zip(self.raw_data_paths, self.indices_paths):
            raw_data = np.load(raw_data_path)
            raw_signal = raw_data["signal"]
            anomaly_label = raw_data["anomaly_label"]

            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines = load_jsonl(indices_path)

            self.normed_signal_list.append(normed_signal)
            self.anomaly_label_list.append(anomaly_label)
            self.index_lines_list.append(index_lines)


        self.global_index = []
        for region_id, index_lines in enumerate(self.index_lines_list):
            for i in range(len(index_lines)):
                self.global_index.append((region_id, i))


    def __getitem__(self, index):
        which_list, which_index = self.global_index[index]

        ts_start = self.index_lines_list[which_list][which_index]["start"]
        ts_end = self.index_lines_list[which_list][which_index]["end"]
        ts_length = ts_end - ts_start

        infill_length = self.min_infill_length + math.floor(torch.sigmoid(torch.rand(1)).item() * (self.max_infill_length - self.min_infill_length))

        relative_anomaly_start = random.randint(0, ts_length - infill_length)
        relative_anomaly_end = relative_anomaly_start + infill_length


        if self.one_channel:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end, :1])
        else:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end])

        # normalize each slide window
        # scaler = MinMaxScaler()
        # signal = scaler.fit_transform(signal)

        # ===== missing signals =====
        missing_signals = torch.zeros(self.max_infill_length, signal.shape[-1])
        missing_signals[:infill_length] = signal[relative_anomaly_start:relative_anomaly_end]
        missing_signals_mask = torch.zeros(self.max_infill_length)
        missing_signals_mask[:infill_length] = 1

        anomaly_label = torch.from_numpy(self.anomaly_label_list[which_list][ts_start:ts_end])
        T = anomaly_label.shape[0]
        # ===== attention mask =====
        # normal + target anomaly are visible
        context_mask = torch.zeros(T, dtype=torch.long)
        context_mask[anomaly_label == 0] = 1
        context_mask[relative_anomaly_start:relative_anomaly_end] = 1

        # ===== noise mask =====
        noise_mask = torch.zeros(T, dtype=torch.long)
        noise_mask[relative_anomaly_start:relative_anomaly_end] = 1


        return {
            'signals': signal,
            'missing_signals': missing_signals,
            'missing_signals_mask': missing_signals_mask,
            'attn_mask': context_mask,
            'noise_mask': noise_mask,
        }


    def __len__(self):
        return len(self.global_index)


class ImputationNormalECGDatasetForSample(Dataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            event_labels_paths,
            seq_len,
            one_channel,
            min_infill_length,
            max_infill_length,
    ):
        super(ImputationNormalECGDatasetForSample, self).__init__()
        self.seq_len = seq_len
        self.one_channel = one_channel
        self.raw_data_paths = raw_data_paths
        self.indices_paths = indices_paths
        self.event_labels_paths = event_labels_paths

        self.min_infill_length = min_infill_length
        self.max_infill_length = max_infill_length

        self.normed_signal_list = []
        self.index_lines_list = []
        self.anomaly_label_list = []
        self.event_label_list = []
        for raw_data_path, indices_path, event_labels_path in zip(self.raw_data_paths, self.indices_paths, self.event_labels_paths):
            raw_data = np.load(raw_data_path)
            raw_signal = raw_data["signal"]
            anomaly_label = raw_data["anomaly_label"]

            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines = load_jsonl(indices_path)

            self.normed_signal_list.append(normed_signal)
            self.anomaly_label_list.append(anomaly_label)
            self.index_lines_list.append(index_lines)
            self.event_label_list.append(np.load(event_labels_path))


        self.global_index = []
        for region_id, index_lines in enumerate(self.index_lines_list):
            for i in range(len(index_lines)):
                self.global_index.append((region_id, i))


    def __getitem__(self, index):
        which_list, which_index = self.global_index[index]

        ts_start = self.index_lines_list[which_list][which_index]["start"]
        ts_end = self.index_lines_list[which_list][which_index]["end"]
        ts_length = ts_end - ts_start


        event_pos = self.event_label_list[which_list]
        event_pos = event_pos[(event_pos >= ts_start) & (event_pos < ts_end)]
        event_pos = event_pos - ts_start  # relative positions
        start_event_idx = random.randint(0, len(event_pos) - 3)
        relative_anomaly_start = int(event_pos[start_event_idx])

        possible_infill_lengths = []
        for end_event_idx in range(start_event_idx, len(event_pos)):
            length_tmp = event_pos[end_event_idx] - event_pos[start_event_idx]
            if 0.9 * self.min_infill_length < length_tmp < self.max_infill_length:
                possible_infill_lengths.append(length_tmp)


        infill_length = random.choice(possible_infill_lengths)
        relative_anomaly_end = relative_anomaly_start + infill_length


        if self.one_channel:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end, :1])
        else:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end])

        # normalize each slide window
        # scaler = MinMaxScaler()
        # signal = scaler.fit_transform(signal)

        # ===== missing signals =====
        missing_signals = torch.zeros(self.max_infill_length, signal.shape[-1])
        missing_signals[:infill_length] = signal[relative_anomaly_start:relative_anomaly_end]
        missing_signals_mask = torch.zeros(self.max_infill_length)
        missing_signals_mask[:infill_length] = 1

        anomaly_label = torch.from_numpy(self.anomaly_label_list[which_list][ts_start:ts_end])
        T = anomaly_label.shape[0]
        # ===== attention mask =====
        # normal + target anomaly are visible
        context_mask = torch.zeros(T, dtype=torch.long)
        context_mask[anomaly_label == 0] = 1
        context_mask[relative_anomaly_start:relative_anomaly_end] = 1

        # ===== noise mask =====
        noise_mask = torch.zeros(T, dtype=torch.long)
        noise_mask[relative_anomaly_start:relative_anomaly_end] = 1


        return {
            'signals': signal,
            'missing_signals': missing_signals,
            'missing_signals_mask': missing_signals_mask,
            'attn_mask': context_mask,
            'noise_mask': noise_mask,
        }


    def __len__(self):
        return len(self.global_index)



class NoContextNormalECGDataset(Dataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            seq_len,
            one_channel,
            min_infill_length,
            max_infill_length,
    ):
        super(NoContextNormalECGDataset, self).__init__()
        self.seq_len = seq_len
        self.one_channel = one_channel

        self.raw_data_paths = raw_data_paths
        self.indices_paths = indices_paths

        self.min_infill_length = min_infill_length
        self.max_infill_length = max_infill_length

        self.normed_signal_list = []
        self.index_lines_list = []
        # self.anomaly_label_list = []
        for raw_data_path, indices_path in zip(self.raw_data_paths, self.indices_paths):
            raw_data = np.load(raw_data_path)
            raw_signal = raw_data["signal"]
            # anomaly_label = raw_data["anomaly_label"]

            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines = load_jsonl(indices_path)
            self.normed_signal_list.append(normed_signal)
            self.index_lines_list.append(index_lines)
            # self.anomaly_label_list.append(anomaly_label)

        self.global_index = []
        for region_id, index_lines in enumerate(self.index_lines_list):
            for i in range(len(index_lines)):
                self.global_index.append((region_id, i))

    def __getitem__(self, index):

        which_list, which_index = self.global_index[index]

        ts_start = self.index_lines_list[which_list][which_index]["start"]
        ts_end = self.index_lines_list[which_list][which_index]["end"]
        ts_length = ts_end - ts_start

        if "source_file" in self.index_lines_list[which_list][which_index].keys():
            infill_length = random.randint(self.min_infill_length, self.max_infill_length)
            relative_anomaly_start = random.randint(0, ts_length - infill_length)
            relative_anomaly_end = relative_anomaly_start + infill_length
        else:
            infill_length = ts_length
            relative_anomaly_start = 0
            relative_anomaly_end = relative_anomaly_start + infill_length



        if self.one_channel:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end, :1])
        else:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end])

        # normalize each slide window
        # scaler = MinMaxScaler()
        # signal = scaler.fit_transform(signal)

        # ===== missing signals =====
        missing_signals = torch.zeros(self.max_infill_length, signal.shape[-1])
        missing_signals[:infill_length] = signal[relative_anomaly_start:relative_anomaly_end]

        context_mask = torch.zeros(self.max_infill_length, dtype=torch.long)
        context_mask[:infill_length] = 1
        return {
            'signals': missing_signals,
            'attn_mask': context_mask,
        }


    def __len__(self):
        return len(self.global_index)



class NoContextAnomalyECGDataset(Dataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            seq_len,
            one_channel,
    ):
        super(NoContextAnomalyECGDataset, self).__init__()
        self.seq_len = seq_len
        self.one_channel = one_channel
        self.raw_data_paths = raw_data_paths
        self.indices_paths = indices_paths

        self.normed_signal_list = []
        self.index_lines_list = []
        self.anomaly_label_list = []

        for raw_data_path, indices_path in zip(self.raw_data_paths, self.indices_paths):
            raw_data = np.load(raw_data_path)
            raw_signal = raw_data["signal"]
            anomaly_label = raw_data["anomaly_label"]

            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines = load_jsonl(indices_path)

            self.normed_signal_list.append(normed_signal)
            self.anomaly_label_list.append(anomaly_label)
            self.index_lines_list.append(index_lines)

        self.global_index = []
        for region_id, index_lines in enumerate(self.index_lines_list):
            for i in range(len(index_lines)):
                self.global_index.append((region_id, i))

    def __getitem__(self, index):

        which_list, which_index = self.global_index[index]

        ts_start = self.index_lines_list[which_list][which_index]["start"]
        ts_end = self.index_lines_list[which_list][which_index]["end"]

        # ts_start = self.index_lines[index]['start']
        # ts_end = self.index_lines[index]['end']
        ts_length = ts_end - ts_start

        data = self.normed_signal_list[which_list][ts_start:ts_end]
        signal = torch.zeros(self.seq_len, data.shape[-1])
        signal[:ts_length] = torch.from_numpy(data)

        if self.one_channel:
            signal = signal[:, :1]

        # normalize each slide window
        # scaler = MinMaxScaler()
        # signal = scaler.fit_transform(signal)

        context_mask = torch.zeros(self.seq_len, dtype=torch.long)
        context_mask[:ts_length] = 1
        return {
            'signals': signal,
            'attn_mask': context_mask,
        }


    def __len__(self):
        return len(self.global_index)


def pad_collate_fn(batch):
    """
    batch: list of Tensor [L_i, C]
    """
    lengths = torch.tensor([x.shape[0] for x in batch], dtype=torch.long)
    max_len = lengths.max().item()
    C = batch[0].shape[-1]

    padded = torch.zeros(len(batch), max_len, C)

    for i, x in enumerate(batch):
        padded[i, :x.shape[0]] = x

    return padded, lengths


if __name__ == "__main__":

    # dataset = NoContextAnomalyECGDataset(
    #     raw_data_paths=["./raw_data/213.npz"],
    #     indices_paths=["./indices/slide_windows_213npz/anomaly_segments.jsonl"],
    #     seq_len=1000,
    #     one_channel=0,
    # )

    # dataset = PredictionECGDataset(
    #     raw_data_paths=["./raw_data/106.npz"],
    #     indices_paths=["./indices/slide_windows_106npz/train/anomaly_segments_with_prototype_train.jsonl"],
    #     seq_len=1200,
    #     one_channel=False,
    #     pre_context_length=400,
    #     max_infill_length=800,
    # )

    # dataset = PredictionECGDataset(
    #     raw_data_paths=["./raw_data_qtdb/sel233.npz"],
    #     indices_paths=["./indices_qtdb/slide_windows_sel233npz/V_segments_train.jsonl"],
    #     seq_len=800,
    #     one_channel=False,
    #     pre_context_length=350,
    #     max_infill_length=450,
    # )

    # dataset = PredictionECGDataset(
    #     raw_data_paths=["./raw_data_svdb/859.npz"],
    #     indices_paths=["./indices_svdb/slide_windows_859npz/V_segments_train.jsonl"],
    #     seq_len=800,
    #     one_channel=False,
    #     pre_context_length=440,
    #     max_infill_length=360,
    # )

    dataset = PredictionECGDataset(
        raw_data_paths=["./raw_data_incart/I30.npz"],
        indices_paths=["./indices_incart/slide_windows_I30npz/V_segments_train.jsonl"],
        seq_len=1000,
        one_channel=False,
        pre_context_length=400,
        max_infill_length=600,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        # plt.plot(batch['signals'][0,:,0], label="Channel 0")
        # plt.plot(batch['signals'][0,:,1], label="Channel 1")
        # plt.plot(batch['attn_mask'][0], label="attn mask")
        # plt.plot(batch['noise_mask'][0], label="noise mask")
        # plt.legend()
        # plt.show(block=False)
        # plt.pause(2.0)  # 停留 2 秒
        # plt.close()
        print("123")
        # break



    # raw_data = np.load("./raw_data/106.npz")
    # raw_signal = raw_data["signal"]
    # anomaly_label = raw_data["anomaly_label"]