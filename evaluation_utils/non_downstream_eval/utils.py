from .discriminator_score import calculate_discriminator_score
from .predictive_score import calculate_predictive_score
from .context_fid import calculate_Context_FID
from .correlational import calculate_correlational
import json
import numpy as np



def calculate_four_metrics(
    ori_data,
    fake_data,
    device,
    num_runs,
    save_path
):

    discriminator_scores = []
    predictive_scores = []
    context_fids = []
    correlations = []

    for i in range(num_runs):
        print(f"[Run {i+1}/{num_runs}]")

        disc_score = calculate_discriminator_score(
            ori_data,
            fake_data[:, i],
            device=device,
            lr=5e-4,
            max_epochs=200,
            batch_size=32,
        )

        pred_score = calculate_predictive_score(
            ori_data,
            fake_data[:, i],
            device=device,
            lr=5e-4,
            max_epochs=2000,
            batch_size=32,
        )

        ctx_fid = calculate_Context_FID(
            ori_data,
            fake_data[:, i],
            device=device,
        )

        corr = calculate_correlational(
            ori_data,
            fake_data[:, i],
        )

        discriminator_scores.append(disc_score)
        predictive_scores.append(pred_score)
        context_fids.append(ctx_fid)
        correlations.append(corr)

    # ---------
    # statistics
    # ---------
    def stat(x):
        x = np.array(x, dtype=np.float64)
        return {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)),  # sample std
            "raw": x.tolist(),
        }

    results = {
        "discriminator_score": stat(discriminator_scores),
        "predictive_score": stat(predictive_scores),
        "context_fid": stat(context_fids),
        "correlational": stat(correlations),
    }

    # ---------
    # save
    # ---------
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Metrics saved to: {save_path}")

    return results
