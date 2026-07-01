# datasetdiff.py
# Dual-Branch Mode:
# X_past   = past(t-Input~t-1) using X_PAST_COLS
# X_future = future(t~t+N-1) using X_FUTURE_COLS
# y        = target(t~t+N-1), support:
#            1) PM2.5_obs
#            2) PM2.5_diff = PM2.5_obs - PM2.5
#            3) O3_obs
#            4) O3_diff = O3_obs - O3

import argparse
import json
import numpy as np
import pandas as pd
import calendar
import torch
from torch.utils.data import TensorDataset

import config as config


def arg():
    parser = argparse.ArgumentParser(description="Dataset Generation (Dual-Branch Mode)")
    parser.add_argument('--Input_timestep', type=int, default=config.Input_timesteps, help='Input_timestep (past length).')
    parser.add_argument('--Output_timestep', type=int, default=config.Output_timesteps, help='Output_timestep (future length).')
    parser.add_argument('--site_sets', type=str, default=str(config.site_sets), help='site_sets, e.g. "[68]".')
    parser.add_argument('--output_dir', type=str, default='results', help='Output folder name.')
    args = parser.parse_args()

    with open(f"./{args.output_dir}/Arglog.txt", 'w') as f:
        f.write(f"Dataset Generation (Dual-Branch Mode): Input={args.Input_timestep}, Output={args.Output_timestep}\n")
        f.write(f"X_PAST_COLS={config.X_PAST_COLS}\n")
        f.write(f"X_FUTURE_COLS={config.X_FUTURE_COLS}\n")
        f.write(f"PAST_INPUT_DIM={config.PAST_INPUT_DIM}\n")
        f.write(f"FUTURE_INPUT_DIM={config.FUTURE_INPUT_DIM}\n")
        f.write(f"Y_COL={config.Y_COL}\n")
    return args


def get_obs_and_sim_col(y_col):
    """
    根據 Y_COL 判斷觀測欄位與模擬欄位。

    例如：
    PM2.5_diff -> PM2.5_obs, PM2.5
    O3_diff    -> O3_obs, O3
    PM2.5_obs  -> PM2.5_obs, None
    O3_obs     -> O3_obs, None
    """
    if y_col.endswith("_diff"):
        pollutant = y_col.replace("_diff", "")
        obs_col = f"{pollutant}_obs"
        sim_col = pollutant
        return obs_col, sim_col

    if y_col.endswith("_obs"):
        return y_col, None

    raise ValueError(
        f"不支援的 Y_COL={y_col}。"
        f"目前僅支援 '*_obs' 或 '*_diff'，例如 PM2.5_obs、PM2.5_diff、O3_obs、O3_diff。"
    )


def data_gen(site_sets, time_steps, output_timesteps, output_dir):
    OrigData_path = config.OrigData_path
    _fileName_sets = config._fileName_sets

    HIST_DIR = getattr(config, "HIST_DIR", "./HIST_DIR")
    year_of_data = getattr(config, "year_of_data", 2016)

    X_PAST_COLS = config.X_PAST_COLS
    X_FUTURE_COLS = config.X_FUTURE_COLS
    Y_COL = config.Y_COL

    OBS_COL, SIM_COL = get_obs_and_sim_col(Y_COL)

    need_x = set(config.Need_element_X)
    need_y = set(config.Need_element_Y)

    extra_needed = set()

    if Y_COL.endswith("_diff"):
        extra_needed.update([OBS_COL, SIM_COL])

    if Y_COL.endswith("_obs"):
        extra_needed.update([OBS_COL])

    needed = need_x.union(need_y).union(extra_needed)

    past_input_dim = len(X_PAST_COLS)
    future_input_dim = len(X_FUTURE_COLS)

    for _s in site_sets:
        _dict = {}

        # ---- 1) read full year (mon 1..12) ----
        for _base in _fileName_sets:
            for mon_i in range(1, 13):
                format_mon = str(mon_i).zfill(2)
                filename = list(_base)
                filename.insert(4, format_mon)
                filename = ''.join(filename)

                try:
                    with open(f'{OrigData_path}/{filename}.txt', 'r') as f:
                        for line in f:
                            _temp = line.strip().split()
                            if len(_temp) < 5:
                                continue

                            site_id = int(_temp[0])
                            cls = _temp[2]

                            if site_id == int(_s) and (cls in needed):
                                aft_4 = _temp[4:]
                                vals = [float(v) for v in aft_4]

                                if cls not in _dict:
                                    _dict[cls] = vals
                                else:
                                    _dict[cls] += vals
                except FileNotFoundError:
                    continue

        pd_data = pd.DataFrame(_dict)

        # ---- 2) clean & interpolate ----

        # 2-1) 缺值標記：-99.9 -> NaN
        pd_data.replace(-99.9, np.nan, inplace=True)

        # 2-2) 其他欄位照原本 linear interpolate
        # 避開 OBS_COL，避免觀測欄位長段缺值被先補掉
        other_cols = [c for c in pd_data.columns if c != OBS_COL]
        if other_cols:
            pd_data[other_cols] = pd_data[other_cols].interpolate(method="linear", inplace=False)

        # 2-3) 觀測欄位 OBS_COL 的分段補值
        if OBS_COL in pd_data.columns:
            s = pd_data[OBS_COL].copy()

            # (a) 先補短洞：最多連續 2 筆 NaN 才會被內插補上；不補頭尾
            s = s.interpolate(method="linear", limit=2, limit_area="inside")

            # (b) 若仍有 NaN，用 HIST 進行同月日小時平均補值
            if s.isna().any():
                days_in_year = 366 if calendar.isleap(year_of_data) else 365
                expected_hours = days_in_year * 24

                if len(s) != expected_hours:
                    raise ValueError(
                        f"[Site {_s}] 資料長度 ({len(s)}) 與 year_of_data={year_of_data} 的全年小時數 ({expected_hours}) 不一致，"
                        f"無法安全進行 HIST 補值。"
                    )

                time_axis = pd.date_range(
                    start=f"{year_of_data}-01-01 00:00:00",
                    periods=len(s),
                    freq="h"
                )

                hist_path = f"{HIST_DIR}/station{int(_s)}.xlsx"
                try:
                    hist_df = pd.read_excel(hist_path)
                except Exception as e:
                    raise ValueError(f"[Site {_s}] 讀取 HIST 失敗: {hist_path} ({e})")

                if hist_df.shape[1] < 2:
                    raise ValueError(f"[Site {_s}] HIST 格式不正確（至少要有 time + 年份欄位）: {hist_path}")

                time_col = hist_df.columns[0]
                year_cols = list(hist_df.columns[1:])

                hist_df[time_col] = pd.to_datetime(hist_df[time_col], errors="coerce")
                if hist_df[time_col].isna().any():
                    raise ValueError(f"[Site {_s}] HIST time 欄位存在無法解析的時間值: {hist_path}")

                hist_key = list(zip(
                    hist_df[time_col].dt.month,
                    hist_df[time_col].dt.day,
                    hist_df[time_col].dt.hour
                ))

                hist_vals = hist_df[year_cols].apply(pd.to_numeric, errors="coerce")
                hist_mean = hist_vals.mean(axis=1, skipna=True)
                hist_map = dict(zip(hist_key, hist_mean.values))

                nan_idx = np.flatnonzero(s.isna().values)
                for idx in nan_idx:
                    dt = time_axis[idx]
                    key = (dt.month, dt.day, dt.hour)
                    v = hist_map.get(key, np.nan)
                    if pd.notna(v):
                        s.iat[idx] = float(v)

            # (c) HIST 後若仍 NaN，最後用線性內插補
            if s.isna().any():
                s = s.interpolate(method="linear", limit_area="inside")

            pd_data[OBS_COL] = s

        # ---- 3) build target column according to Y_COL ----
        if Y_COL.endswith("_diff"):
            if OBS_COL not in pd_data.columns or SIM_COL not in pd_data.columns:
                raise ValueError(
                    f"pd_data 缺少 '{OBS_COL}' 或 '{SIM_COL}'，無法計算 {Y_COL}。"
                )

            pd_data[Y_COL] = pd_data[OBS_COL].values - pd_data[SIM_COL].values

        # ---- 4) sanity check columns ----
        missing_past = [c for c in X_PAST_COLS if c not in pd_data.columns]
        missing_future = [c for c in X_FUTURE_COLS if c not in pd_data.columns]

        if missing_past or missing_future:
            raise ValueError(
                f"資料缺欄位：missing_past={missing_past}, missing_future={missing_future}。"
                f"請確認 txt 是否包含這些 cls，或調整 X_PAST_COLS / X_FUTURE_COLS。"
            )

        if Y_COL not in pd_data.columns:
            raise ValueError(
                f"目標欄位 {Y_COL} 不存在於 pd_data 中。"
                f"請確認 config.Y_COL 設定與前處理邏輯一致。"
            )

        X_past_list, X_future_list, y_list = [], [], []

        # ---- 5) make samples (t0 = i) ----
        # X_past   : [i-time_steps, i)
        # X_future : [i, i+output_timesteps)
        # y        : [i, i+output_timesteps)
        for i in range(time_steps, len(pd_data) - output_timesteps + 1):
            x_past = pd_data.iloc[i - time_steps:i][X_PAST_COLS].values.astype(np.float32)
            x_future = pd_data.iloc[i:i + output_timesteps][X_FUTURE_COLS].values.astype(np.float32)
            y = pd_data.iloc[i:i + output_timesteps][[Y_COL]].values.astype(np.float32)

            X_past_list.append(x_past)
            X_future_list.append(x_future)
            y_list.append(y)

        X_past_array = np.asarray(X_past_list, dtype=np.float32)
        X_future_array = np.asarray(X_future_list, dtype=np.float32)
        y_array = np.asarray(y_list, dtype=np.float32)

        # ---- 6) shape checks ----
        if X_past_array.ndim != 3 or X_past_array.shape[1] != time_steps or X_past_array.shape[2] != past_input_dim:
            raise ValueError(
                f"X_past shape mismatch: got {X_past_array.shape}, "
                f"expected (*, {time_steps}, {past_input_dim})."
            )

        if X_future_array.ndim != 3 or X_future_array.shape[1] != output_timesteps or X_future_array.shape[2] != future_input_dim:
            raise ValueError(
                f"X_future shape mismatch: got {X_future_array.shape}, "
                f"expected (*, {output_timesteps}, {future_input_dim})."
            )

        if y_array.ndim != 3 or y_array.shape[1] != output_timesteps or y_array.shape[2] != 1:
            raise ValueError(
                f"y shape mismatch: got {y_array.shape}, expected (*, {output_timesteps}, 1)."
            )

        print(
            f"[Site {_s}] "
            f"X_past shape: {X_past_array.shape}  "
            f"X_future shape: {X_future_array.shape}  "
            f"y shape: {y_array.shape}  "
            f"target={Y_COL}  "
            f"obs_col={OBS_COL}  "
            f"sim_col={SIM_COL}"
        )

        X_past_tensor = torch.tensor(X_past_array, dtype=torch.float32)
        X_future_tensor = torch.tensor(X_future_array, dtype=torch.float32)
        y_tensor = torch.tensor(y_array, dtype=torch.float32)

        dataset = TensorDataset(X_past_tensor, X_future_tensor, y_tensor)

        save_path = f"./{output_dir}/{_s}_diff_training_data.pt"
        torch.save(dataset, save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    args = arg()
    data_gen(
        site_sets=json.loads(args.site_sets),
        time_steps=args.Input_timestep,
        output_timesteps=args.Output_timestep,
        output_dir=args.output_dir
    )