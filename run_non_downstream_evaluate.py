from evaluation_utils import calculate_four_metrics
import argparse
import torch







parser = argparse.ArgumentParser(description="parameters for non-downstream evaluation")

"""time series general parameters"""
parser.add_argument("--fake_data_path", type=str, required=True)
parser.add_argument("--real_data_path", type=str, required=True)
parser.add_argument("--num_iters", type=int, required=True)
parser.add_argument("--save_dir", type=str, required=True)

args = parser.parse_args()
ori_data = torch.load(args.real_data_path)
ori_data = ori_data["all_samples"]
fake_data = torch.load(args.fake_data_path) # todo: this is a dict
calculate_four_metrics(
    ori_data,
    fake_data,
    device="cuda",
    num_runs=5,
    save_path=args.save_dir
)
