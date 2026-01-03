from transformers import GPT2Tokenizer, GPT2Model
import torch.nn as nn
import argparse
from GPT4TS import GPT4TSModel
import torch

def get_args():
    import argparse

    parser = argparse.ArgumentParser(description='TimesNet')

    # ========================
    # Task settings
    # ========================
    parser.add_argument('--task_name', type=str, default='short_term_forecast',
                        choices=['long_term_forecast', 'short_term_forecast',
                                 'imputation', 'anomaly_detection', 'classification'],
                        help='task name')

    # ========================
    # Data settings
    # ========================
    parser.add_argument('--seq_len', type=int, default=100,
                        help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=100,
                        help='prediction length')
    parser.add_argument('--enc_in', type=int, default=1,
                        help='number of input variables (channels)')
    parser.add_argument('--c_out', type=int, default=1,
                        help='number of output variables')
    parser.add_argument('--freq', type=str, default='h',
                        help='frequency for time features')

    # ========================
    # Patch settings
    # ========================
    parser.add_argument('--patch_size', type=int, default=1,
                        help='patch length')
    parser.add_argument('--stride', type=int, default=1,
                        help='patch stride')

    # ========================
    # Embedding & model dims
    # ========================
    parser.add_argument('--d_model', type=int, default=512,
                        help='embedding dimension')
    parser.add_argument('--d_ff', type=int, default=512,
                        help='hidden dimension used from GPT output')
    parser.add_argument('--embed', type=str, default='fixed',
                        choices=['fixed', 'learned', 'timeF'],
                        help='time embedding type')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')

    # ========================
    # GPT-2 settings
    # ========================
    parser.add_argument('--gpt_layers', type=int, default=6,
                        help='number of GPT-2 layers to use')
    parser.add_argument('--mlp', type=int, default=0,
                        choices=[0, 1],
                        help='whether to finetune GPT MLP layers')
    parser.add_argument('--ln', type=int, default=1,
                        choices=[0, 1],
                        help='whether to use layer norm')
    parser.add_argument('--use_gpu', type=bool, default=False,
                        help='use gpu or not')

    # ========================
    # Classification task
    # ========================
    parser.add_argument('--num_class', type=int, default=2,
                        help='number of classes for classification')

    args = parser.parse_args()
    return args


import torch
import torch.nn.functional as F

def patchify_time_series(
    x: torch.Tensor,
    patch_size: int,
    stride: int,
    padding: bool = False
):
    """
    Patchify a time series.

    Args:
        x: Tensor of shape (B, T, C)
        patch_size: int, length of each patch
        stride: int, stride between patches
        padding: bool, whether to pad at the end to cover all timesteps

    Returns:
        x_patch: Tensor of shape (B, N, patch_size * C)
    """
    B, T, C = x.shape

    # (B, T, C) -> (B, C, T)
    x = x.permute(0, 2, 1)

    if padding:
        pad_len = (stride - (T - patch_size) % stride) % stride
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode="replicate")

    # unfold: (B, C, T) -> (B, C, N, patch_size)
    x = x.unfold(dimension=-1, size=patch_size, step=stride)

    # (B, C, N, P) -> (B, N, P, C)
    x = x.permute(0, 2, 3, 1)

    # (B, N, P, C) -> (B, N, P*C)
    x = x.reshape(B, x.shape[1], patch_size * C)

    return x

if __name__ == '__main__':
    args = get_args()
    model = GPT4TSModel(args)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: shape={tuple(param.shape)}")

    # model = GPT2Model.from_pretrained('gpt2')
    # print(model)
    input_ts = torch.randn(2, args.seq_len, args.enc_in)

    out = model(input_ts)