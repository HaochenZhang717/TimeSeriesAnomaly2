from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        "task_name": "classification",
        "n_channels": 1,
        "num_class": 2
    },
)
model.init()

print(model)

