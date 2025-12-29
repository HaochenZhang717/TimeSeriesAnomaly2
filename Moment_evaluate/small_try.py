from momentfm import MOMENTPipeline
import torch



model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        "task_name": "embedding",
        # "n_channels": 1,
        # "num_class": 2
    },
)
model.init()
input = torch.randn(1, 2, 512)
y = model(x_enc=input)
print(y.shape)
print(model)

