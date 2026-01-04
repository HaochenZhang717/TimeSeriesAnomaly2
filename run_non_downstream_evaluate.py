from evaluation_utils import calculate_four_metrics
import argparse
import torch
import json
import torch
import torch.nn.functional as F
import os


parser = argparse.ArgumentParser(description="parameters for non-downstream evaluation")

"""time series general parameters"""
parser.add_argument("--samples_path", type=str, required=True)
# parser.add_argument("--fake_data_path", type=str, required=True)
# parser.add_argument("--real_data_path", type=str, required=True)
# parser.add_argument("--num_iters", type=int, required=True)
parser.add_argument("--save_dir", type=str, required=True)

args = parser.parse_args()
# ori_data = torch.load(args.real_data_path)
# ori_data = ori_data["all_samples"].cuda()
# fake_data = torch.load(args.fake_data_path) # todo: this is a dict
# fake_data = fake_data["all_samples"].cuda()
# print(fake_data.keys())
# breakpoint()
# calculate_four_metrics(
#     ori_data,
#     fake_data,
#     device="cuda",
#     num_runs=5,
#     save_path=args.save_dir
# )




ori_data = torch.load(args.samples_path)
real = ori_data["all_reals"]        # (B, T, C)
fake = ori_data["all_samples"]      # (B, N, T, C)
mask = ori_data["all_labels"]

mse_list = []
mae_list = []

num_samples = fake.shape[1]
mse_loss = torch.nn.MSELoss(reduction="none")
mae_loss = torch.nn.L1Loss(reduction="none")

for i in range(num_samples):
    pred = fake[:, i]   # (B, T, C)
    label = real


    mse = mse_loss(pred, label).mean(-1)
    mse = mse.sum() / mask.sum()
    mae = F.l1_loss(pred, label).mean(-1)
    mae = mae.sum() / mask.sum()

    mse_list.append(mse.item())
    mae_list.append(mae.item())

    # print(f"[sample {i}] MSE={mse.item():.6f}, MAE={mae.item():.6f}")

# --------
# statistics
# --------
mse_mean = float(torch.tensor(mse_list).mean())
mse_std  = float(torch.tensor(mse_list).std(unbiased=True))

mae_mean = float(torch.tensor(mae_list).mean())
mae_std  = float(torch.tensor(mae_list).std(unbiased=True))

results = {
    "metric": "MSE/MAE",
    "num_samples": num_samples,
    "mse": {
        "mean": mse_mean,
        "std": mse_std,
        "raw": mse_list,
    },
    "mae": {
        "mean": mae_mean,
        "std": mae_std,
        "raw": mae_list,
    },
}
print("mean mse:", mse_mean)
print("mean mae:", mae_mean)
save_path = args.save_dir
dir_path = os.path.dirname(save_path)
os.makedirs(dir_path, exist_ok=True)


with open(save_path, "w") as f:
    f.write(json.dumps(results) + "\n")

print(f"Saved to {save_path}")


