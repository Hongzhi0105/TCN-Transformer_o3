# inference.py

import argparse
import json
import os
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader, Subset

import config as config


def arg():
    parser = argparse.ArgumentParser(description="Dual-Branch inference")
    parser.add_argument('--Input_timestep', type=int, default=config.Input_timesteps)
    parser.add_argument('--Output_timestep', type=int, default=config.Output_timesteps)
    parser.add_argument('--Avoid_interval', type=int, default=7)
    parser.add_argument('--site_sets', type=str, default=str(config.site_sets))
    parser.add_argument('--output_dir', type=str, default='results')
    return parser.parse_args()


# =========================================================
# 🔧 Activation（與 train.py 完全一致）
# =========================================================
def get_fusion_activation(name):
    if name == "ReLU":
        return nn.ReLU()
    elif name == "GELU":
        return nn.GELU()
    elif name == "LeakyReLU":
        return nn.LeakyReLU()
    elif name == "Tanh":
        return nn.Tanh()
    elif name == "PReLU":
        return nn.PReLU(num_parameters=config.FUSION_PRELU_NUM_PARAMETERS)
    else:
        raise ValueError(f"Unsupported FUSION_ACTIVATION: {name}")


# =========================================================
# 原始模型結構（與 train.py 保持一致）
# =========================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TCN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(pad)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # self.relu1 = nn.Identity()
        # self.relu1 = nn.ReLU()
        self.relu1 = nn.PReLU(out_channels)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(pad)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # self.relu2 = nn.Identity()
        # self.relu2 = nn.ReLU()
        self.relu2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.chomp1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.chomp2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class Transformer_encoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )

    def forward(self, x):
        x = x.permute(2, 0, 1)   # (B, C, L) -> (L, B, C)
        x = self.layer(x)
        return x.permute(1, 2, 0)   # (L, B, C) -> (B, C, L)


class Encoder(nn.Module):
    def __init__(self, input_dim, return_sequence=False):
        super().__init__()
        self.return_sequence = return_sequence

        self.tcn_layers = nn.ModuleList()
        in_channels = input_dim

        for i, out_channels in enumerate(config.TCN_CHANNELS):
            self.tcn_layers.append(
                TCN_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.KERNEL_SIZE,
                    dilation=2 ** i
                )
            )
            in_channels = out_channels

        self.transformer = Transformer_encoder(
            config.TRANSFORMER_D_MODEL,
            config.TRANSFORMER_NHEAD,
            config.TRANSFORMER_DIM_FEEDFORWARD,
            config.TRANSFORMER_DROPOUT
        )

    def forward(self, x):
        for layer in self.tcn_layers:
            x = layer(x)

        x = self.transformer(x)

        if self.return_sequence:
            return x
        else:
            return x[:, :, -1]


# =========================================================
# 🔥 DualBranchModel（與 train.py 完全同步）
# 1024 -> 512 -> Act -> 256 -> Act -> 1
# =========================================================
class DualBranchModel(nn.Module):
    def __init__(self, output_timesteps):
        super().__init__()

        self.output_timesteps = output_timesteps

        self.past_encoder = Encoder(
            input_dim=config.PAST_INPUT_DIM,
            return_sequence=False
        )

        self.future_encoder = Encoder(
            input_dim=config.FUTURE_INPUT_DIM,
            return_sequence=True
        )

        fusion_dim = config.FUSION_INPUT_DIM
        fusion_hidden_dim_1 = config.FUSION_HIDDEN_DIM
        fusion_hidden_dim_2 = config.FUSION_HIDDEN_DIM // 2

        if fusion_hidden_dim_2 < 1:
            raise ValueError(
                f"Invalid second fusion hidden dim: {fusion_hidden_dim_2}. "
                f"Please check config.FUSION_HIDDEN_DIM={config.FUSION_HIDDEN_DIM}."
            )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim_1),
            get_fusion_activation(config.FUSION_ACTIVATION),
            nn.Linear(fusion_hidden_dim_1, fusion_hidden_dim_2),
            get_fusion_activation(config.FUSION_ACTIVATION),
            nn.Linear(fusion_hidden_dim_2, 1)
        )

    def forward(self, x_past, x_future):
        # x_past   : (B, C_past, L_past)
        # x_future : (B, C_future, L_future)

        past_feat = self.past_encoder(x_past)        # (B, C)
        future_feat = self.future_encoder(x_future)  # (B, C, N)

        if future_feat.size(-1) != self.output_timesteps:
            raise RuntimeError(
                f"Future length mismatch: got {future_feat.size(-1)}, "
                f"expected {self.output_timesteps}"
            )

        future_feat = future_feat.permute(0, 2, 1)   # (B, N, C)
        past_feat = past_feat.unsqueeze(1).expand(-1, self.output_timesteps, -1)  # (B, N, C)

        fusion = torch.cat([past_feat, future_feat], dim=2)   # (B, N, 2C)

        out = self.fusion_mlp(fusion).squeeze(-1)   # (B, N)

        return out


# =========================================================
# 🔧 Inference
# =========================================================
def inference(model, dataloader, device):
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for x_past, x_future, y in dataloader:
            x_past = x_past.permute(0, 2, 1).to(device)
            x_future = x_future.permute(0, 2, 1).to(device)
            y = y.squeeze(-1).to(device)

            output = model(x_past, x_future)

            preds.append(output.cpu())
            trues.append(y.cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    return preds.numpy(), trues.numpy()


# =========================================================
# 🔧 CSV輸出
# =========================================================
def save_csv(pred, true, save_path):
    N = pred.shape[1]

    columns_pred = [f'Prediction_t+{i+1}' for i in range(N)]
    columns_true = [f'Truth_t+{i+1}' for i in range(N)]

    df = pd.DataFrame(
        data=list(map(lambda x: list(x[0]) + list(x[1]), zip(pred, true))),
        columns=columns_pred + columns_true
    )

    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")


# =========================================================
# 🔧 MAIN
# =========================================================
if __name__ == "__main__":
    args = arg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    site_list = json.loads(args.site_sets)

    for site in site_list:
        process_path = f"./{args.output_dir}/{site}_ratio_process.pt"
        avoid_path = f"./{args.output_dir}/{site}_ratio_avoid.pt"

        process_dataset = torch.load(process_path, weights_only=False)
        process_loader = DataLoader(
            process_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )

        model = DualBranchModel(args.Output_timestep).to(device)

        model_path = f"./{args.output_dir}/ratio_diff_best_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))

        print("Loaded model:", model_path)

        # process
        pred, true = inference(model, process_loader, device)
        save_csv(pred, true, f"./{args.output_dir}/{site}_ratio_process_predict.csv")

        # avoid（若有）
        if os.path.exists(avoid_path):
            avoid_dataset = torch.load(avoid_path, weights_only=False)
            if len(avoid_dataset) > 0:
                avoid_loader = DataLoader(
                    avoid_dataset,
                    batch_size=config.BATCH_SIZE,
                    shuffle=False
                )
                pred, true = inference(model, avoid_loader, device)
                save_csv(pred, true, f"./{args.output_dir}/{site}_ratio_avoid_predict.csv")