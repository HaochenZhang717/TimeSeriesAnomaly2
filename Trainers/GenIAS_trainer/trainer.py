import torch
import wandb
import os
from tqdm import tqdm


class GenIAS_Trainer(object):
    def __init__(
            self,optimizer, scheduler, model, train_loader,
            val_loader, max_epochs, device, save_dir,
            wandb_project_name, wandb_run_name, grad_clip_norm,
            early_stop, patience
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.device = device
        self.save_dir = save_dir
        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name
        self.grad_clip_norm = grad_clip_norm
        self.early_stop = early_stop
        self.patience = patience

    def train(self, config):

        # 初始化 wandb
        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config=config
        )

        os.makedirs(self.save_dir, exist_ok=True)

        best_val_loss = float("inf")
        no_improve_epochs = 0
        global_steps = 0

        for epoch in range(self.max_epochs):
            total_loss = reconstruction_loss = perturbation_loss = kl_loss = 0
            tr_seen = 0
            self.model.train()
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):

                # X_occluded = batch["signal_random_occluded"].to(self.device)
                X_normal = batch["orig_signal"].to(self.device)
                noise_mask = batch["noise_mask"].to(self.device)
                X_occluded = X_normal * noise_mask.unsqueeze(-1)

                self.optimizer.zero_grad()
                loss, recon_loss, perturb_loss, kl = self.model(X_occluded, X_normal, noise_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                total_loss += loss.item()
                reconstruction_loss += recon_loss.item()
                perturbation_loss += perturb_loss.item()
                kl_loss += kl.item()
                tr_seen += 1
                global_steps += 1

            train_total_avg = total_loss / tr_seen
            train_recon_avg = reconstruction_loss / tr_seen
            train_perturb_avg = perturbation_loss / tr_seen
            train_kl_avg = kl_loss / tr_seen


            """evaluation"""
            self.model.eval()
            with torch.no_grad():
                val_total, val_recon, val_perturb, val_kl, val_seen = 0, 0, 0, 0
                for batch in self.val_loader:
                    # X_occluded = batch["signal_random_occluded"].to(self.device)
                    X_normal = batch["orig_signal"].to(self.device)
                    noise_mask = batch["noise_mask"].to(self.device)
                    X_occluded = X_normal * noise_mask.unsqueeze(-1)

                    with torch.no_grad():
                        loss, recon_loss, perturb_loss, kl = self.model(X_occluded, X_normal, noise_mask)

                    val_total += loss.item()
                    val_recon += recon_loss.item()
                    val_perturb += perturb_loss.item()
                    val_kl += kl.item()
                    val_seen += 1

                val_total /= val_seen
                val_recon /= val_seen
                val_perturb /= val_seen
                val_kl /= val_seen

                wandb.log({
                    "train/total_loss": train_total_avg,
                    "train/recon_loss": train_recon_avg,
                    "train/perturb_loss": train_perturb_avg,
                    "train/kl_loss": train_kl_avg,
                    "val/total_loss": val_total,
                    "val/perturb_loss": val_perturb,
                    "val/recon_loss": val_recon,
                    "val/kl_loss": val_kl,
                    "epoch": epoch,
                    "step": global_steps,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

                if self.early_stop == "true":
                    if val_total < best_val_loss:
                        best_val_loss = val_total
                        no_improve_epochs = 0
                        torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                    else:
                        no_improve_epochs += 1

                    if no_improve_epochs >= self.patience:
                        print(f"⛔ Early stopping triggered at Step {global_steps}.")
                        break
                else:
                    torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")

                self.scheduler.step(val_total)

        wandb.finish()


