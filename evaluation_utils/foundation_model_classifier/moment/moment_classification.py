# from momentfm.data.ptbxl_classification_dataset import PTBXL_dataset
from momentfm import MOMENTPipeline
from momentfm.models.statistical_classifiers import fit_svm
from torch.utils.data import TensorDataset
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
import json
import argparse
from argparse import Namespace
import random
import numpy as np
import os
import pdb
from torch import nn
import numpy as np


PRETRAINED_MODEL_NAME = {
    'large': "AutonLab/MOMENT-1-large",
    'base': "AutonLab/MOMENT-1-base",
    'small': "AutonLab/MOMENT-1-small"
}



def control_randomness(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




class PatchToTimeHead(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        n_channels=2,
        patch_size=8,
        hidden_dim=256,
    ):
        super().__init__()
        self.patch_size = patch_size
        in_dim = embed_dim * n_channels

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # patch-level logit
        )

    def forward(self, patch_embeds):
        """
        patch_embeds: [B, C, Np, D]
        return:       [B, T]
        """
        # print(f"patch embeds: {patch_embeds.shape}")
        B, C, Np, D = patch_embeds.shape

        # [B, Np, C*D]
        x = patch_embeds.permute(0, 2, 1, 3).reshape(B, Np, C * D)
        # print(f"x.shape: {x.shape}")
        # [B, Np, 1]
        patch_logits = self.mlp(x)

        # [B, Np]
        patch_logits = patch_logits.squeeze(-1)
        patch_logits = patch_logits.unsqueeze(2)

        # [B, Np, patch_size, 2]
        patch_logits = patch_logits.repeat(1, 1, self.patch_size, 1)

        # [B, T, 2]
        logits = patch_logits.reshape(B, Np * self.patch_size, 2)

        return logits.permute(0,2,1)





class PTBXL_Trainer:
    def __init__(
        self,
        real_data,
        real_labels,
        gen_data,
        gen_labels,
        model_name: str,
        one_channel: bool,
    ):
        args = get_default_args()
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # initialize ptbxl classification dataset
        train_signal = gen_data
        if one_channel:
            train_signal = train_signal[:,:,:1]
        train_label = gen_labels
        train_signal = torch.nn.functional.interpolate(
            train_signal.permute(0,2,1), size=args.seq_len,
            mode="linear", align_corners=False
        )
        train_label = train_label.unsqueeze(1)  # [B, 1, T]
        # breakpoint()
        train_label = torch.nn.functional.interpolate(
            train_label.float(),  # interpolate 只能用 float
            size=args.seq_len,
            mode="nearest"
        )
        train_label = train_label.squeeze(1).long()  # [B, T]

        train_data = TensorDataset(train_signal, train_label)





        test_signal = real_data
        if one_channel:
            test_signal = test_signal[:,:,:1]
        test_label = real_labels

        # adjust size
        test_signal = torch.nn.functional.interpolate(
            test_signal.permute(0,2,1), size=args.seq_len,
            mode="linear", align_corners=False
        )
        test_label = test_label.unsqueeze(1)  # [B, 1, T]
        test_label = torch.nn.functional.interpolate(
            test_label.float(),  # interpolate 只能用 float
            size=args.seq_len,
            mode="nearest"
        )
        test_label = test_label.squeeze(1).long()  # [B, T]

        test_data = TensorDataset(test_signal, test_label)


        self.train_dataset = train_data
        self.val_dataset = test_data
        self.test_dataset = test_data


        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False)



        self.backbone = MOMENTPipeline.from_pretrained(
            PRETRAINED_MODEL_NAME[model_name],
            model_kwargs={
                'task_name': 'embedding',
            },
        )
        self.backbone.init()
        print('Model initialized, training mode: ', self.args.mode)
        self.head = PatchToTimeHead(
            embed_dim=self.backbone.encoder.block[-1].layer[-1].DenseReluDense.wo.out_features,
            n_channels=train_signal.shape[1],
            patch_size=8,
            hidden_dim=256,
        )
        # using cross entropy loss for classification
        self.criterion = torch.nn.CrossEntropyLoss()


        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=self.args.init_lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.max_lr,
                                                             total_steps=self.args.epochs * len(
                                                                 self.train_dataloader))


        self.log_file = open(f'log_{self.args.mode}.txt', 'w')
        self.log_file.write(f'PTBXL classification training, mode: {self.args.mode}\n')

    def get_embeddings(self, dataloader: DataLoader):
        '''
        labels: [num_samples]
        embeddings: [num_samples x d_model]
        '''
        embeddings, labels = [], []

        with torch.no_grad():
            for batch_x, batch_labels in tqdm(dataloader, total=len(dataloader)):
                # [batch_size x 12 x 512]
                batch_x = batch_x.to(self.device).float()
                # [batch_size x num_patches x d_model (=1024)]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and
                                                                                torch.cuda.get_device_capability()[
                                                                                    0] >= 8 else torch.float32):
                    output = self.model(x_enc=batch_x, reduction=self.args.reduction)
                    # mean over patches dimension, [batch_size x d_model]
                embedding = output.embeddings.mean(dim=1)
                embeddings.append(embedding.detach().cpu().numpy())
                labels.append(batch_labels)

        embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
        return embeddings, labels

    def get_timeseries(self, dataloader: DataLoader, agg='mean'):
        '''
        mean: average over all channels, result in [1 x seq_len] for each time-series
        channel: concat all channels, result in [1 x seq_len * num_channels] for each time-series

        labels: [num_samples]
        ts: [num_samples x seq_len] or [num_samples x seq_len * num_channels]
        '''
        ts, labels = [], []

        with torch.no_grad():
            for batch_x, batch_labels in tqdm(dataloader, total=len(dataloader)):
                # [batch_size x 12 x 512]
                if agg == 'mean':
                    batch_x = batch_x.mean(dim=1)
                    ts.append(batch_x.detach().cpu().numpy())
                elif agg == 'channel':
                    ts.append(batch_x.view(batch_x.size(0), -1).detach().cpu().numpy())
                labels.append(batch_labels)

        ts, labels = np.concatenate(ts), np.concatenate(labels)
        return ts, labels

    def train(self):
        for epoch in range(self.args.epochs):

            print(f'Epoch {epoch + 1}/{self.args.epochs}')
            self.log_file.write(f'Epoch {epoch + 1}/{self.args.epochs}\n')
            self.epoch = epoch + 1

            if self.args.mode == 'linear_probing':
                self.train_epoch_lp()
                self.evaluate_epoch()

            elif self.args.mode == 'full_finetuning':
                self.train_epoch_ft()
                self.evaluate_epoch()

            # break after training SVM, only need one 'epoch'
            elif self.args.mode == 'unsupervised_representation_learning':
                self.train_ul()
                break

            elif self.args.mode == 'svm':
                self.train_svm()
                break

            else:
                raise ValueError(
                    'Invalid mode, please choose svm, linear_probing, full_finetuning, or unsupervised_representation_learning')

    #####################################training loops#############################################
    def train_epoch_lp(self):
        '''
        Train only classification head
        '''
        self.backbone.to(self.device)
        self.backbone.eval()
        self.head.to(self.device)
        self.head.train()

        losses = []

        for batch_x, batch_labels in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            self.optimizer.zero_grad()
            batch_x = batch_x.to(self.device).float()
            batch_labels = batch_labels.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and
                                                                            torch.cuda.get_device_capability()[
                                                                                0] >= 8 else torch.float32):
                backbone_output = self.backbone(x_enc=batch_x, reduction="none")
                # print(backbone_output.embeddings.shape)
                # breakpoint()
                output_logits = self.head(backbone_output.embeddings)

                # breakpoint()
                loss = self.criterion(output_logits, batch_labels)
                # print(output.embeddings.shape)
                # output = self.model(x_enc=batch_x.permute(0,2,1), reduction=self.args.reduction)
                # loss = self.criterion(output.logits, batch_labels)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print('Train loss: ', avg_loss)
        self.log_file.write(f'Train loss: {avg_loss}\n')

    def train_epoch_ft(self):
        '''
        Train encoder and classification head (with accelerate enabled)
        '''
        self.model.to(self.device)
        self.model.train()
        losses = []

        for batch_x, batch_labels in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            self.optimizer.zero_grad()
            batch_x = batch_x.to(self.device).float()
            batch_labels = batch_labels.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and
                                                                            torch.cuda.get_device_capability()[
                                                                                0] >= 8 else torch.float32):
                output = self.model(x_enc=batch_x, reduction=self.args.reduction)
                loss = self.criterion(output.logits, batch_labels)
                losses.append(loss.item())
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.scheduler.step()

        avg_loss = np.mean(losses)
        print('Train loss: ', avg_loss)
        self.log_file.write(f'Train loss: {avg_loss}\n')

    def train_ul(self):
        '''
        Train SVM on top of MOMENT embeddings
        '''
        self.model.eval()
        self.model.to(self.device)

        # extract embeddings and label
        train_embeddings, train_labels = self.get_embeddings(self.train_dataloader)
        # print('embedding shape: ', train_embeddings.shape)
        # print('label shape: ', train_labels.shape)

        # fit statistical classifier
        self.clf = fit_svm(features=train_embeddings, y=train_labels)
        train_accuracy = self.clf.score(train_embeddings, train_labels)
        print('Train accuracy: ', train_accuracy)
        self.log_file.write(f'Train accuracy: {train_accuracy}\n')

    def train_svm(self):
        '''
        Train SVM on top of timeseries data
        '''
        train_embeddings, train_labels = self.get_timeseries(self.train_dataloader, agg=self.args.agg)
        self.clf = fit_svm(features=train_embeddings, y=train_labels)
        train_accuracy = self.clf.score(train_embeddings, train_labels)
        print('Train accuracy: ', train_accuracy)
        self.log_file.write(f'Train accuracy: {train_accuracy}\n')

    #####################################training loops#################################################

    #####################################evaluate loops#################################################
    def test(self):
        if self.args.mode == 'unsupervised_representation_learning':
            test_embeddings, test_labels = self.get_embeddings(self.test_dataloader)
            test_accuracy = self.clf.score(test_embeddings, test_labels)
            print('Test accuracy: ', test_accuracy)
            self.log_file.write(f'Test accuracy: {test_accuracy}\n')

        elif self.args.mode == 'linear_probing' or self.args.mode == 'full_finetuning':
            return self.evaluate_epoch(phase='test')

        elif self.args.mode == 'svm':
            test_embeddings, test_labels = self.get_timeseries(self.test_dataloader, agg=self.args.agg)
            test_accuracy = self.clf.score(test_embeddings, test_labels)
            print('Test accuracy: ', test_accuracy)
            self.log_file.write(f'Test accuracy: {test_accuracy}\n')

        else:
            raise ValueError(
                'Invalid mode, please choose linear_probing, full_finetuning, or unsupervised_representation_learning')

    def evaluate_epoch(self, phase='val'):
        if phase == 'val':
            dataloader = self.val_dataloader
        elif phase == 'test':
            dataloader = self.test_dataloader
        else:
            raise ValueError('Invalid phase, please choose val or test')

        self.backbone.eval()
        self.backbone.to(self.device)

        self.head.eval()
        self.head.to(self.device)

        total_loss, total_correct = 0, 0
        total_samples = 0
        TP = FP = FN = TN = 0

        with torch.no_grad():
            for batch_x, batch_labels in tqdm(dataloader, total=len(dataloader)):
                batch_x = batch_x.to(self.device).float()
                batch_labels = batch_labels.to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and
                                                                                torch.cuda.get_device_capability()[
                                                                           0] >= 8 else torch.float32):
                    backbone_output = self.backbone(x_enc=batch_x, reduction="none")
                    output_logits = self.head(backbone_output.embeddings)
                    loss = self.criterion(output_logits, batch_labels)

                    # output = self.model(x_enc=batch_x)
                    # loss = self.criterion(output.logits, batch_labels)
                total_loss += loss.item()

                preds = output_logits.argmax(dim=1)

                total_correct += (preds == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

                # confusion matrix update
                TP += ((preds == 1) & (batch_labels == 1)).sum().item()
                FP += ((preds == 1) & (batch_labels == 0)).sum().item()
                FN += ((preds == 0) & (batch_labels == 1)).sum().item()
                TN += ((preds == 0) & (batch_labels == 0)).sum().item()

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / len(dataloader.dataset)
        print(f'{phase} loss: {avg_loss}, {phase} accuracy: {accuracy}')
        # self.log_file.write(f'{phase} loss: {avg_loss}, {phase} accuracy: {accuracy}, {phase} F1: {f1} \n')
        self.log_file.write(
            f"{phase} loss: {avg_loss}, "
            f"accuracy: {accuracy}, "
            f"precision: {precision}, "
            f"recall: {recall}, "
            f"f1: {f1}\n"
        )
        return f1
    #####################################evaluate loops#################################################

    def save_checkpoint(self):
        if self.args.mode in ['svm', 'unsupervised_representation_learning']:
            raise ValueError('No checkpoint to save for SVM or unsupervised learning, as no training was done')

        path = self.args.output_path

        # mkdir if not exist
        if not os.path.exists(path):
            os.makedirs(path)

        # save parameter that requires grad
        torch.save(self.model.state_dict(), os.path.join(path, 'MOMENT_Classification.pth'))
        print('Model saved at ', path)


def get_default_args():

    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--mode', type=str, default='linear_probing',
                        help='choose from linear_probing, full_finetuning, unsupervised_representation_learning')
    parser.add_argument('--init_lr', type=float, default=1e-6)
    parser.add_argument('--max_lr', type=float, default=1e-4)
    parser.add_argument('--agg', type=str, default='channel',
                        help='aggregation method for timeseries data for svm training, choose from mean or channel')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lora', action='store_true', help='enable LoRA')
    parser.add_argument('--reduction', type=str, default='concat',
                        help='reduction method for MOMENT embeddings, choose from mean or max')
    parser.add_argument('--output_path', type=str, help='path to save trained model and logs')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='sequence length for each sample, currently only support 512 for MOMENT')
    parser.add_argument('--load_cache', type=bool, default=True, help='whether to load cached dataset')
    args = parser.parse_args([])
    return args


def run_moment_evaluate(
        real_data,
        real_labels,
        gen_data,
        gen_labels,
        model_name,
        one_channel,
        output_path
    ):
    all_results = []
    for run_id in range(5):
        print(f"\n========== Run {run_id} ==========\n")
        seed = run_id
        control_randomness(seed)

        trainer = PTBXL_Trainer(
            real_data, real_labels,
            gen_data, gen_labels,
            model_name=model_name,
            one_channel=one_channel)

        trainer.train()
        metrics = trainer.test()  # 建议 test() return metric
        all_results.append(float(metrics))

    all_results = np.array(all_results)

    results_dict = {
        "metric": "F1",
        "num_runs": len(all_results),
        "values": all_results.tolist(),
        "mean": float(all_results.mean()),
        "std": float(all_results.std()),
    }

    # ---- save ----
    save_path = os.path.join(output_path, "moment_evaluation_results.json;")
    with open(save_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"Results saved to {save_path}")


if __name__ == '__main__':
    pass


    # control_randomness(args.seed)
    #
    # trainer = PTBXL_Trainer(args)
    # trainer.train()
    # f1 = trainer.test()
    # trainer.save_checkpoint()

