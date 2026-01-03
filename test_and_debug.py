import torch
from evaluation_utils import calculate_discriminator_score
from evaluation_utils import calculate_predictive_score
from evaluation_utils import calculate_Context_FID
from evaluation_utils import calculate_correlational

batch_size = 64
seq_len = 100
ts_dim = 2

ori_data = torch.randn(batch_size, seq_len, ts_dim)
fake_data = torch.randn(batch_size, seq_len, ts_dim)

calculate_discriminator_score(
    ori_data,
    fake_data,
    device="cpu",
    lr=5e-4,
    max_epochs=200,
    batch_size=4,
)

# calculate_predictive_score(
#     ori_data,
#     fake_data,
#     device="cpu",
#     lr=5e-4,
#     max_epochs=2000,
#     batch_size=64,
# )

calculate_Context_FID(ori_data, fake_data)

loss = calculate_correlational(ori_data, fake_data)
print(loss)