# train.py
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset

import config as config


def arg():
    parser = argparse.ArgumentParser(description="Dual-Branch TCN+Transformer training")
    parser.add_argument('--Input_timestep', type=int, default=config.Input_timesteps)
    parser.add_argument('--Output_timestep', type=int, default=config.Output_timesteps)
    parser.add_argument('--Avoid_interval', type=int, default=7)
    parser.add_argument('--site_sets', type=str, default=str(config.site_sets))
    parser.add_argument('--output_dir', type=str, default='results')
    return parser.parse_args()


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
    elif name == "Identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported FUSION_ACTIVATION: {name}")


# ===============================
# Causal crop helper
# ===============================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


# ===============================
# TCN BLOCK (length-preserving causal)
# ===============================
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
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out


# ===============================
# Transformer Encoder
# ===============================
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
        # x: (B, C, L)
        x = x.permute(2, 0, 1)    # (L, B, C)
        out = self.layer(x)
        return out.permute(1, 2, 0)   # (B, C, L)


# ===============================
# Encoder (TCN + Transformer)
# ===============================
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
            d_model=config.TRANSFORMER_D_MODEL,
            nhead=config.TRANSFORMER_NHEAD,
            dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=config.TRANSFORMER_DROPOUT
        )

    def forward(self, x):
        # x: (B, C, L)
        for layer in self.tcn_layers:
            x = layer(x)

        x = self.transformer(x)   # (B, C, L)

        if self.return_sequence:
            return x              # 保留整段 sequence
        else:
            return x[:, :, -1]    # 只取最後 timestep


# ===============================
# Dual Branch Model (Point-to-Point, Scheme A)
# past  : global vector from t-72 ~ t-1
# future: 72 timestep hidden from t ~ t+71
# ===============================
class DualBranchModel(nn.Module):
    def __init__(self, output_timesteps):
        super().__init__()

        self.output_timesteps = output_timesteps

        # Past branch -> single global hidden
        self.past_encoder = Encoder(
            input_dim=config.PAST_INPUT_DIM,
            return_sequence=False
        )

        # Future branch -> keep full future sequence
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

        # 修改 fusion head：
        # 1024 -> 512 -> Act -> 256 -> Act -> 1
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

        # (B, C)
        past_feat = self.past_encoder(x_past)

        # (B, C, L_future_encoded)
        future_feat = self.future_encoder(x_future)

        # 保險檢查：future encoder 的輸出長度應與 output_timesteps 一致
        if future_feat.size(-1) != self.output_timesteps:
            raise RuntimeError(
                f"Future branch encoded length mismatch: "
                f"got {future_feat.size(-1)}, expected {self.output_timesteps}. "
                f"Please check TCN length preservation."
            )

        # (B, C, N) -> (B, N, C)
        future_feat = future_feat.permute(0, 2, 1)

        # (B, C) -> (B, 1, C) -> (B, N, C)
        past_feat = past_feat.unsqueeze(1).expand(-1, self.output_timesteps, -1)

        # (B, N, C) + (B, N, C) -> (B, N, 2C)
        fusion = torch.cat([past_feat, future_feat], dim=2)

        # (B, N, 2C) -> (B, N, 1) -> (B, N)
        out = self.fusion_mlp(fusion).squeeze(-1)

        return out


# ===============================
# TRAIN FUNCTION
# ===============================
def train(site_sets, time_steps, output_timesteps, output_dir, train_loader, valid_loader, RatioInterval='ratio'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DualBranchModel(output_timesteps).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=50
    )

    num_epochs = 1000

    train_losses = []
    valid_losses = []

    best_valid_rmse = float('inf')
    best_model_path = f'./{output_dir}/{RatioInterval}_diff_best_model.pth'

    print("Best model path:", best_model_path)

    for epoch in range(num_epochs):
        model.train()
        train_total = 0.0

        for X_past_batch, X_future_batch, y_batch in train_loader:
            X_past_batch = X_past_batch.permute(0, 2, 1).to(device)      # (B, L, C) -> (B, C, L)
            X_future_batch = X_future_batch.permute(0, 2, 1).to(device)  # (B, L, C) -> (B, C, L)
            y_batch = y_batch.squeeze(-1).to(device)                     # (B, N, 1) -> (B, N)

            optimizer.zero_grad()

            output = model(X_past_batch, X_future_batch)                 # (B, N)
            loss = criterion(output, y_batch)

            loss.backward()
            optimizer.step()

            train_total += torch.sqrt(loss).item()

        train_rmse = train_total / max(1, len(train_loader))
        train_losses.append(train_rmse)

        model.eval()
        valid_total = 0.0

        with torch.no_grad():
            for X_past_batch, X_future_batch, y_batch in valid_loader:
                X_past_batch = X_past_batch.permute(0, 2, 1).to(device)
                X_future_batch = X_future_batch.permute(0, 2, 1).to(device)
                y_batch = y_batch.squeeze(-1).to(device)

                output = model(X_past_batch, X_future_batch)
                loss = criterion(output, y_batch)

                valid_total += torch.sqrt(loss).item()

        valid_rmse = valid_total / max(1, len(valid_loader))
        valid_losses.append(valid_rmse)

        scheduler.step(valid_rmse)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train RMSE: {train_rmse:.4f}  "
            f"Valid RMSE: {valid_rmse:.4f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.8f}"
        )

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            torch.save(model.state_dict(), best_model_path)
            print("Valid RMSE improved, saving best model.")

        pd.DataFrame({
            'Train_RMSE': train_losses,
            'Valid_RMSE': valid_losses
        }).to_csv(
            f'./{output_dir}/{RatioInterval}_diff_best_model_loss.csv',
            index=False
        )

    print("Best Valid RMSE:", best_valid_rmse)
    print("Best model saved to:", best_model_path)


# ===============================
# DATA SPLIT + SAVE PROCESS/AVOID PT
# ===============================
def Ratio_data_split(site_sets, output_dir, ratio=0.8, avoid_interval=1, input_step=24):
    all_train_subsets = []
    all_valid_subsets = []

    # 2016 leap year
    days_per_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hours_per_month = [d * 24 for d in days_per_month]

    month_hour_start = [0]
    for h in hours_per_month:
        month_hour_start.append(month_hour_start[-1] + h)

    os.makedirs(output_dir, exist_ok=True)

    for _s in site_sets:
        dataset = torch.load(f"./{output_dir}/{_s}_diff_training_data.pt", weights_only=False)
        N = len(dataset)

        all_indices_set = set(range(N))

        # avoid_interval = 0 時：
        # process = 全部資料
        # avoid   = 空資料
        process_indices = sorted(list(all_indices_set))
        avoid_save_indices = []

        if avoid_interval != 0:
            m = avoid_interval
            ms = month_hour_start[m - 1]
            me = month_hour_start[m]

            # t0 = d + input_step
            # process/train 排除區：d ∈ [ms-input_step, me)
            d_excl_start = max(0, ms - input_step)
            d_excl_end = min(N, me)
            exclude_for_train = set(range(d_excl_start, d_excl_end))

            # avoid 預測集：d ∈ [ms, me-input_step)
            d_avoid_start = max(0, ms)
            d_avoid_end = min(N, me - input_step)
            avoid_for_predict = set(range(d_avoid_start, d_avoid_end))

            process_indices = sorted(list(all_indices_set - exclude_for_train))
            avoid_save_indices = sorted(list(avoid_for_predict))

        process_subset = Subset(dataset, process_indices)
        avoid_subset = Subset(dataset, avoid_save_indices)

        process_save_path = f"./{output_dir}/{_s}_ratio_process.pt"
        avoid_save_path = f"./{output_dir}/{_s}_ratio_avoid.pt"

        torch.save(process_subset, process_save_path)
        torch.save(avoid_subset, avoid_save_path)

        print(f"[Site {_s}] Saved process dataset: {process_save_path} (N={len(process_subset)})")
        print(f"[Site {_s}] Saved avoid dataset:   {avoid_save_path} (N={len(avoid_subset)})")

        train_size = int(ratio * len(process_subset))
        valid_size = len(process_subset) - train_size

        current_train_subset, current_valid_subset = random_split(
            process_subset,
            [train_size, valid_size]
        )

        all_train_subsets.append(current_train_subset)
        all_valid_subsets.append(current_valid_subset)

    combined_train_dataset = ConcatDataset(all_train_subsets)
    combined_valid_dataset = ConcatDataset(all_valid_subsets)

    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    valid_loader = DataLoader(
        combined_valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    print("Train_sets:", len(combined_train_dataset))
    print("Valid_sets:", len(combined_valid_dataset))

    return train_loader, valid_loader


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    args = arg()
    site_list = json.loads(args.site_sets)

    train_loader, valid_loader = Ratio_data_split(
        site_sets=site_list,
        ratio=0.8,
        output_dir=args.output_dir,
        avoid_interval=args.Avoid_interval,
        input_step=args.Input_timestep
    )

    train(
        site_sets=site_list,
        time_steps=args.Input_timestep,
        output_timesteps=args.Output_timestep,
        output_dir=args.output_dir,
        train_loader=train_loader,
        valid_loader=valid_loader,
        RatioInterval="ratio"
    )