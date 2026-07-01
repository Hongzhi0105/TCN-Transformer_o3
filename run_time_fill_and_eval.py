"""整合「時間填補」與「性能評估」的執行腳本。

這支程式的主要用途：
1. 讓使用者一開始選擇要跑 O3 或 PM2.5。
2. 讓使用者選擇要一次處理哪些月份與測站。
3. 先檢查每個預測 CSV 是否有正確的 time 欄位，沒有就自動補上。
4. 接著用 base_data.csv 的 CMAQ 與觀測值計算每月性能指標。
5. 如果同一方案/測站的 1、4、7、10 月都已處理，才會輸出年度評估。

資料夾命名假設：
- 月份方案資料夾需長得像 TCN_ReLU-JAN-42_basecase 或 TCN_PReLU-JAN-42_K2。
- 程式會從資料夾名稱解析月份 JAN/APR/JUL/OCT 與測站編號 42/43/44/45。
- 預測檔預設尋找 *_ratio_avoid_predict.csv。

路徑使用方式：
- O3 預設使用本腳本所在資料夾，也就是 TCN_Transformer_o3_v2。
- PM2.5 預設使用相對路徑 ../TCN_Transformer_pm。
- 可以用 --work-dir 與 --base-data 改成其他相對路徑，方便未來套用到新方案。
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# 全域設定與命名規則
# =============================================================================

# 本研究目前評估四個代表月份；年度評估也必須同時具備這四個月份。
MONTH_ABBR = {1: "JAN", 4: "APR", 7: "JUL", 10: "OCT"}
ANNUAL_MONTHS = set(MONTH_ABBR)
ABBR_MONTH = {abbr: month for month, abbr in MONTH_ABBR.items()}

# 從方案資料夾名稱解析：
# prefix: 方案名稱前段，例如 TCN_ReLU 或 TCN_PReLU
# abbr:   月份縮寫，例如 JAN
# site:   測站編號，例如 42
# suffix: 方案名稱後段，例如 _basecase 或 _K2
MONTH_FOLDER_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<abbr>JAN|APR|JUL|OCT)-(?P<site>\d+)(?P<suffix>.*)$",
    flags=re.IGNORECASE,
)

# 預測欄位命名，例如 Prediction_t+1 到 Prediction_t+72。
PRED_COL_RE = re.compile(r"^Prediction_t\+(\d+)$")

# 評估輸出的指標欄位。順序會直接反映在輸出的 CSV 內。
METRICS_COLS = [
    "R2",
    "IOA",
    "MAE",
    "RMSE",
    "THR(%)",
    "HHR(%)",
    "FAR(%)",
    "Over_Est(%)",
    "Under_Est(%)",
]

# 原始資料中代表缺值的數值，讀取 base_data.csv 後會統一轉成 NaN。
MISSING_VALUES = [-99.9, -999, -999.0]


# =============================================================================
# 資料結構
# =============================================================================

@dataclass(frozen=True)
class PollutantSpec:
    """污染物相關設定。

    high_threshold 用來計算 HHR/FAR 的高污染門檻。
    aqi_bins 用來把數值轉成台灣 AQI 等級，再計算 THR。
    """

    name: str
    high_threshold: float
    aqi_bins: tuple[float, ...]


@dataclass(frozen=True)
class RuntimeConfig:
    """一次執行所需的固定設定。

    這些值會從 config.py、命令列參數或互動選單整理後集中放在這裡，
    後續函式就不用一直傳很多零散參數。
    """

    year: int
    input_timesteps: int
    output_timesteps: int
    strict_month_boundary: bool
    model_label: str
    mode_suffix: str
    pollutant: PollutantSpec


@dataclass(frozen=True)
class MonthTarget:
    """代表一個要處理的月份預測檔。

    例如 TCN_ReLU-JAN-42_basecase/42_ratio_avoid_predict.csv 會被整理成：
    month=1, abbr=JAN, site=42, prefix=TCN_ReLU, suffix=_basecase。
    """

    folder: Path
    prediction_csv: Path
    prefix: str
    suffix: str
    abbr: str
    month: int
    site: int

    @property
    def annual_folder_name(self) -> str:
        """由月份資料夾推回年度輸出資料夾名稱。"""

        return f"{self.prefix}-ALL-{self.site}{self.suffix}"

    @property
    def group_key(self) -> tuple[str, int, str]:
        """年度合併時的分組鍵：同一方案、同一測站、同一 suffix 才能合併。"""

        return self.prefix, self.site, self.suffix


@dataclass
class DaySeries:
    """儲存 Day 1/Day 2/Day 3 的預測值、觀測值與 CMAQ baseline。"""

    predict: list[float]
    obs: list[float]
    cmaq: list[float]


def canonical_pollutant(value: str) -> str:
    """把使用者輸入的污染物名稱標準化。

    使用者可能輸入 O3、o3、PM、PM2.5、pm25 等不同寫法。
    後續程式只使用兩個內部代碼：o3 與 pm25，避免到處判斷不同字串。
    """

    key = value.strip().lower().replace("_", "").replace("-", "").replace(".", "")
    if key == "o3":
        return "o3"
    if key in {"pm", "pm25", "pm2", "pm25"}:
        return "pm25"
    raise argparse.ArgumentTypeError("pollutant must be one of: o3, pm, pm2.5")


def pollutant_spec(name: str) -> PollutantSpec:
    """依污染物取得門檻與 AQI 分級設定。

    O3 與 PM2.5 的 AQI 分級及高污染判定門檻不同，
    所以評估前必須先依污染物建立對應設定。
    """

    if name == "o3":
        return PollutantSpec(name="O3", high_threshold=71.0, aqi_bins=(54, 70, 85, 105, 200, 404))
    return PollutantSpec(
        name="PM2.5",
        high_threshold=30.5,
        aqi_bins=(12.4, 30.4, 50.4, 125.4, 225.4, 325.4),
    )


# =============================================================================
# 路徑與 config 輔助函式
# =============================================================================

def display_path(path: Path, base: Path) -> str:
    """把路徑轉成相對於 base 的顯示格式。

    這只影響 print 出來的訊息，不影響實際讀寫檔案。
    儘量顯示相對路徑，讓輸出符合目前專案的使用習慣。
    """

    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path.resolve(), base.resolve()).replace("\\", "/")


def resolve_work_dir(raw_work_dir: str | None, script_dir: Path) -> Path:
    """解析方案工作目錄。

    raw_work_dir 若為 None，代表使用腳本所在資料夾。
    若是相對路徑，固定以 script_dir 為基準，避免從不同終端目錄執行時路徑跑掉。
    """

    if raw_work_dir is None:
        return script_dir.resolve()
    path = Path(raw_work_dir)
    if path.is_absolute():
        return path.resolve()
    return (script_dir / path).resolve()


def resolve_from_base(raw_path: str, base_dir: Path) -> Path:
    """把使用者提供的檔案路徑解析成絕對路徑。

    這裡用在 --base-data 與 --config。
    如果使用者給相對路徑，就以該次評估的 work_dir 為基準。
    """

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config_defaults(work_dir: Path, config_arg: str | None) -> dict[str, int]:
    """從 config.py 讀取預設年份與時間步長。

    若找不到 config.py，或 config.py 讀取失敗，會退回常用預設值：
    year=2016, input_timesteps=72, output_timesteps=72。
    這樣此腳本可獨立執行，不會強制依賴 config.py。
    """

    defaults = {
        "year": 2016,
        "input_timesteps": 72,
        "output_timesteps": 72,
    }
    config_path = resolve_from_base(config_arg, work_dir) if config_arg else work_dir / "config.py"
    if not config_path.exists():
        return defaults

    try:
        spec = importlib.util.spec_from_file_location("_eval_runtime_config", config_path)
        if spec is None or spec.loader is None:
            return defaults
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"[warn] Could not load config defaults from {display_path(config_path, work_dir)}: {exc}")
        return defaults

    defaults["year"] = int(getattr(module, "year_of_data", defaults["year"]))
    defaults["input_timesteps"] = int(getattr(module, "Input_timesteps", defaults["input_timesteps"]))
    defaults["output_timesteps"] = int(getattr(module, "Output_timesteps", defaults["output_timesteps"]))
    return defaults


def find_base_data(
    work_dir: Path,
    script_dir: Path,
    pollutant: str,
    base_data_arg: str | None,
) -> Path:
    """尋找評估用的 base_data.csv。

    優先順序：
    1. 使用者明確指定的 --base-data。
    2. 目前 work_dir 或腳本資料夾下的 base_data.csv。
    3. 依污染物尋找常見資料夾，例如 O3 的 TCN_Transformer_o3_v1，
       或 PM2.5 的 TCN_Transformer_pm。

    base_data.csv 必須包含 time、obs_測站、CMAQ_測站欄位。
    """

    if base_data_arg:
        base_data = resolve_from_base(base_data_arg, work_dir)
        if not base_data.exists():
            raise FileNotFoundError(f"Base data not found: {display_path(base_data, work_dir)}")
        return base_data

    candidates = [work_dir / "base_data.csv", script_dir / "base_data.csv"]
    if pollutant == "o3":
        candidates.extend(
            [
                script_dir.parent / "TCN_Transformer_o3_v1" / "base_data.csv",
                script_dir.parent / "TCN_Transformer_o3" / "base_data.csv",
                work_dir.parent / "TCN_Transformer_o3_v1" / "base_data.csv",
                work_dir.parent / "TCN_Transformer_o3" / "base_data.csv",
            ]
        )
    else:
        candidates.extend(
            [
                script_dir.parent / "TCN_Transformer_pm" / "base_data.csv",
                work_dir.parent / "TCN_Transformer_pm" / "base_data.csv",
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Base data not found. Pass --base-data with a path relative to --work-dir."
    )


# =============================================================================
# 預測檔 time 欄位檢查與填補
# =============================================================================

def expected_time_series(month: int, row_count: int, cfg: RuntimeConfig) -> pd.DatetimeIndex:
    """建立某月份預測檔應該具備的 time 序列。

    訓練/推論設定是使用過去 input_timesteps 小時預測未來 output_timesteps 小時。
    因此 1 月資料的第一筆預測時間不是 1/1 00:00，而是：
    1/1 00:00 + input_timesteps。
    目前 input_timesteps 預設為 72，所以第一筆通常是每月 4 日 00:00。
    """

    month_start = pd.Timestamp(year=cfg.year, month=month, day=1, hour=0)
    start_time = month_start + pd.Timedelta(hours=cfg.input_timesteps)
    return pd.date_range(start=start_time, periods=row_count, freq="h")


def normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """統一 time 欄位名稱。

    舊檔案可能用 Time，大部分新檔用 time。
    評估流程統一使用小寫 time。
    """

    if "Time" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"Time": "time"})
    return df


def has_valid_time(df: pd.DataFrame, month: int, cfg: RuntimeConfig) -> tuple[bool, pd.Series | None]:
    """檢查預測檔是否已經有正確 time 欄位。

    條件包含：
    1. 存在 time 或 Time 欄位。
    2. 所有 time 都能被 pandas 解析成日期時間。
    3. time 序列與 expected_time_series() 產生的序列完全一致。

    回傳 True 時代表不用補時間；False 時代表要重新建立 time 欄位。
    """

    df = normalize_time_column(df)
    if "time" not in df.columns:
        return False, None

    time_values = pd.to_datetime(df["time"], errors="coerce")
    if time_values.isna().any():
        return False, None

    expected = expected_time_series(month, len(df), cfg)
    if len(time_values) == 0:
        return True, time_values

    is_expected = time_values.reset_index(drop=True).equals(pd.Series(expected))
    return is_expected, time_values if is_expected else None


def repair_shifted_time_header(df: pd.DataFrame, cfg: RuntimeConfig) -> pd.DataFrame:
    """修復少數因 time 標頭位移造成的欄位錯位。

    有些 CSV 可能第一欄叫 time，但實際資料沒有 time 值，最後多出一欄全 NaN。
    這通常代表欄位標頭整體往右偏了一格。
    若偵測到這種格式，會移除最後一欄並重建 Prediction/Truth 欄名。
    """

    expected_cols = cfg.output_timesteps * 2
    if "time" not in df.columns or len(df.columns) != expected_cols + 1:
        return df
    if not df.iloc[:, -1].isna().all():
        return df

    repaired = df.iloc[:, :-1].copy()
    repaired.columns = (
        [f"Prediction_t+{idx + 1}" for idx in range(cfg.output_timesteps)]
        + [f"Truth_t+{idx + 1}" for idx in range(cfg.output_timesteps)]
    )
    return repaired


def ensure_time_column(target: MonthTarget, cfg: RuntimeConfig, work_dir: Path) -> bool:
    """確保單一預測 CSV 具備正確 time 欄位。

    若檔案已經有正確 time，僅輸出 [time:keep]。
    若檔案缺少 time 或 time 不正確，會依月份與 input_timesteps 重新補上，
    並直接覆寫原本的 *_ratio_avoid_predict.csv。
    """

    df = pd.read_csv(target.prediction_csv)
    valid, time_values = has_valid_time(df, target.month, cfg)
    label = f"{target.abbr}-{target.site}"
    csv_label = display_path(target.prediction_csv, work_dir)

    # 已有正確 time 時不做資料重排，只在 Time 欄位大小寫需要統一時重寫一次。
    if valid:
        df = normalize_time_column(df)
        if "Time" in df.columns:
            df.to_csv(target.prediction_csv, index=False)
        first = time_values.iloc[0] if time_values is not None and len(time_values) else "empty"
        last = time_values.iloc[-1] if time_values is not None and len(time_values) else "empty"
        print(f"[time:keep] {label}: {csv_label} ({first} -> {last})")
        return False

    # time 不存在或不正確時，重新計算完整時間序列放在第一欄。
    df = normalize_time_column(df)
    df = repair_shifted_time_header(df, cfg)
    df = df.drop(columns=["time", "Time"], errors="ignore")
    time_series = expected_time_series(target.month, len(df), cfg)
    output = pd.concat([pd.DataFrame({"time": time_series}), df.reset_index(drop=True)], axis=1)
    output.to_csv(target.prediction_csv, index=False)

    if len(output):
        print(
            f"[time:fill] {label}: {csv_label} "
            f"({output['time'].iloc[0]} -> {output['time'].iloc[-1]})"
        )
    else:
        print(f"[time:fill] {label}: {csv_label} (empty file)")
    return True


def discover_targets(
    work_dir: Path,
    prediction_glob: str,
    sites: set[int] | None,
    months: set[int] | None,
    scheme_regex: str | None,
) -> list[MonthTarget]:
    """掃描 work_dir，找出這次要處理的預測檔。

    搜尋流程：
    1. 逐一查看 work_dir 底下的資料夾。
    2. 用 MONTH_FOLDER_RE 判斷是否符合「方案-月份-測站-suffix」命名。
    3. 依使用者選的 months、sites、scheme_regex 過濾。
    4. 在符合的資料夾內尋找 prediction_glob，預設是 *_ratio_avoid_predict.csv。

    回傳的 MonthTarget 已經包含月份、測站、資料夾與 CSV 路徑，
    後續流程不用再重新解析資料夾名稱。
    """

    scheme_filter = re.compile(scheme_regex) if scheme_regex else None
    targets: list[MonthTarget] = []

    for folder in sorted(path for path in work_dir.iterdir() if path.is_dir()):
        match = MONTH_FOLDER_RE.match(folder.name)
        if not match:
            continue
        if scheme_filter and not scheme_filter.search(folder.name):
            continue

        abbr = match.group("abbr").upper()
        month = ABBR_MONTH[abbr]
        site = int(match.group("site"))
        if sites is not None and site not in sites:
            continue
        if months is not None and month not in months:
            continue

        prediction_files = sorted(folder.glob(prediction_glob))
        for prediction_csv in prediction_files:
            # 檔名通常以測站開頭，例如 42_ratio_avoid_predict.csv。
            # 若檔名測站與資料夾測站不同，代表資料夾或檔案放錯，直接跳過。
            file_site = prediction_csv.name.split("_", 1)[0]
            if file_site.isdigit() and int(file_site) != site:
                print(
                    f"[skip] Site mismatch: folder={site}, file={file_site}, "
                    f"path={display_path(prediction_csv, work_dir)}"
                )
                continue
            targets.append(
                MonthTarget(
                    folder=folder,
                    prediction_csv=prediction_csv,
                    prefix=match.group("prefix"),
                    suffix=match.group("suffix"),
                    abbr=abbr,
                    month=month,
                    site=site,
                )
            )

    return targets


# =============================================================================
# 評估指標計算
# =============================================================================

def get_aqi_level(value: float, pollutant: PollutantSpec) -> int:
    """把污染物濃度轉成 AQI 等級。

    THR 指標不是直接比較數值，而是比較預測值與觀測值是否落在同一 AQI 等級。
    """

    for idx, upper_bound in enumerate(pollutant.aqi_bins):
        if value <= upper_bound:
            return idx
    return len(pollutant.aqi_bins)


def calc_metrics(pred: Iterable[float], obs: Iterable[float], pollutant: PollutantSpec) -> list[float]:
    """計算單一 Day 的性能指標。

    pred 與 obs 會先轉成 numpy array，並移除 NaN/inf。
    回傳順序必須與 METRICS_COLS 相同，才能正確寫入 CSV。

    指標說明：
    - R2: 相關係數平方。
    - IOA: Index of Agreement。
    - MAE/RMSE: 絕對誤差與均方根誤差。
    - THR: 預測與觀測 AQI 等級相同的比例。
    - HHR: 觀測高污染事件中被預測命中的比例。
    - FAR: 預測為高污染但觀測不是高污染的比例。
    - Over/Under: 預測高於/低於觀測的比例。
    """

    pred_arr = np.asarray(list(pred), dtype=float)
    obs_arr = np.asarray(list(obs), dtype=float)
    valid_mask = np.isfinite(pred_arr) & np.isfinite(obs_arr)
    pred_arr = pred_arr[valid_mask]
    obs_arr = obs_arr[valid_mask]

    if len(pred_arr) == 0:
        return [math.nan] * len(METRICS_COLS)

    rmse = float(np.sqrt(np.mean((pred_arr - obs_arr) ** 2)))
    mae = float(np.mean(np.abs(pred_arr - obs_arr)))

    # 樣本數不足時無法穩定計算相關係數，沿用舊腳本邏輯給 0.0。
    if len(pred_arr) > 1:
        corr = np.corrcoef(pred_arr, obs_arr)[0, 1]
        r2 = float(corr**2) if np.isfinite(corr) else math.nan
    else:
        r2 = 0.0

    obs_mean = float(np.mean(obs_arr))
    ioa_den = np.sum((np.abs(pred_arr - obs_mean) + np.abs(obs_arr - obs_mean)) ** 2)
    ioa = float(1 - (np.sum((pred_arr - obs_arr) ** 2) / ioa_den)) if ioa_den != 0 else 0.0

    # THR 先把連續濃度轉成 AQI 等級，再比較分類是否一致。
    pred_aqi = np.array([get_aqi_level(value, pollutant) for value in pred_arr])
    obs_aqi = np.array([get_aqi_level(value, pollutant) for value in obs_arr])
    thr = float(np.sum(pred_aqi == obs_aqi) / len(obs_arr) * 100)

    # HHR/FAR 依污染物門檻判定高污染事件；若分母為 0 則輸出 NaN。
    pred_high = pred_arr >= pollutant.high_threshold
    obs_high = obs_arr >= pollutant.high_threshold
    hhr = float(np.sum(pred_high & obs_high) / np.sum(obs_high) * 100) if np.sum(obs_high) else math.nan
    far = float(np.sum(pred_high & ~obs_high) / np.sum(pred_high) * 100) if np.sum(pred_high) else math.nan

    over = float(np.sum(pred_arr > obs_arr) / len(obs_arr) * 100)
    under = float(np.sum(pred_arr < obs_arr) / len(obs_arr) * 100)

    return [r2, ioa, mae, rmse, thr, hhr, far, over, under]


def load_base_data(site: int, base_data_path: Path) -> pd.DataFrame:
    """讀取指定測站的 CMAQ 與觀測資料。

    base_data.csv 需包含：
    - time
    - CMAQ_{site}
    - obs_{site}

    讀入後會：
    1. 把 -99.9、-999 等缺值代碼轉成 NaN。
    2. 移除完全空白的尾端列。
    3. 以每小時完整時間軸 reindex，讓缺漏小時變成 NaN。
    4. 回傳欄位統一命名為 CMAQ 與 Obs。
    """

    df = pd.read_csv(base_data_path)
    df.replace(MISSING_VALUES, np.nan, inplace=True)

    if "time" not in df.columns:
        raise ValueError(f"{base_data_path} must contain a time column.")

    parsed_time = pd.to_datetime(df["time"], errors="coerce")
    invalid_time = parsed_time.isna()
    if invalid_time.any():
        # 有些 base_data.csv 尾端會多一列全空白資料。
        # 全空列可以安全移除；若該列有其他資料卻沒有合法 time，代表資料本身錯誤。
        non_time_cols = [col for col in df.columns if col != "time"]
        invalid_rows_have_data = df.loc[invalid_time, non_time_cols].notna().any(axis=1)
        invalid_rows_have_time_text = df.loc[invalid_time, "time"].notna()
        bad_rows = invalid_rows_have_data | invalid_rows_have_time_text
        if bad_rows.any():
            raise ValueError(f"{base_data_path} contains invalid time values.")
        df = df.loc[~invalid_time].copy()
        parsed_time = parsed_time.loc[~invalid_time]

    df["time"] = parsed_time

    df = df.set_index("time").sort_index()

    # 建立完整逐小時時間軸，確保後續 loc[t0:t0+71h] 可以檢查是否湊滿 72 筆。
    full_time_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    df = df.reindex(full_time_idx)

    cmaq_col = f"CMAQ_{site}"
    obs_col = f"obs_{site}"
    if cmaq_col not in df.columns or obs_col not in df.columns:
        raise ValueError(f"base_data.csv missing required columns: {cmaq_col}, {obs_col}")

    return df[[cmaq_col, obs_col]].rename(columns={cmaq_col: "CMAQ", obs_col: "Obs"})


def prediction_values(row: pd.Series, output_timesteps: int) -> np.ndarray:
    """從單一預測列取出未來 output_timesteps 小時的預測差值。

    正常情況下會使用 Prediction_t+1、Prediction_t+2 ... 欄位。
    若舊檔案沒有這種欄名，則退回舊腳本邏輯：
    移除 time 欄後取前 output_timesteps 個數值。
    """

    pred_cols: list[tuple[int, str]] = []
    for col in row.index:
        match = PRED_COL_RE.match(str(col))
        if match:
            pred_cols.append((int(match.group(1)), str(col)))

    if len(pred_cols) >= output_timesteps:
        selected = [col for _, col in sorted(pred_cols)[:output_timesteps]]
        return row[selected].to_numpy(dtype=float)

    values = row.drop(labels=["time", "Time"], errors="ignore").to_numpy(dtype=float)
    if len(values) < output_timesteps:
        raise ValueError(f"Prediction row has {len(values)} values, need {output_timesteps}.")
    return values[:output_timesteps]


def empty_day_series(day_count: int) -> list[DaySeries]:
    """建立 Day 1/Day 2/Day 3 等容器。"""

    return [DaySeries(predict=[], obs=[], cmaq=[]) for _ in range(day_count)]


def process_month_data(target: MonthTarget, base_df: pd.DataFrame, cfg: RuntimeConfig) -> list[DaySeries]:
    """把單月預測差值轉成可評估的 Day 1/2/3 絕對濃度資料。

    重要流程：
    1. 讀取預測 CSV，確認 time 欄位存在且可解析。
    2. 只取每天 00:00 的列作為 72 小時預報起點。
    3. 從 base_data.csv 找出 t0 到 t0+71 的 CMAQ/Obs。
    4. 預測檔中的值是「修正差值」，所以 AI 絕對值 = 預測差值 + CMAQ。
    5. 將 72 小時切成 Day 1、Day 2、Day 3，各自累積預測、觀測與 CMAQ。
    """

    df = pd.read_csv(target.prediction_csv)
    df = normalize_time_column(df)
    if "time" not in df.columns:
        raise ValueError(
            f"{display_path(target.prediction_csv, target.folder.parent)} has no time column. "
            "Run time fill first."
        )

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError(f"{target.prediction_csv} contains invalid time values.")

    day_count = cfg.output_timesteps // 24
    if day_count < 1:
        raise ValueError("output_timesteps must be at least 24 for daily evaluation.")

    day_data = empty_day_series(day_count)

    # 每小時預測列都有 72 小時預報，但評估以每天 00:00 起報的 72 小時為主。
    df_00 = df[(df["time"].dt.hour == 0) & (df["time"].dt.month == target.month)].copy()

    for _, row in df_00.iterrows():
        t0 = row["time"]
        base_window = base_df.loc[t0 : t0 + pd.Timedelta(hours=cfg.output_timesteps - 1)]
        if len(base_window) != cfg.output_timesteps:
            continue

        pred_diff = prediction_values(row, cfg.output_timesteps)
        if not np.isfinite(pred_diff).all():
            continue

        cmaq_values = base_window["CMAQ"].to_numpy(dtype=float)
        obs_values = base_window["Obs"].to_numpy(dtype=float)

        # 模型預測的是 obs-CMAQ 的修正量，因此要加回 CMAQ 才是最終預測濃度。
        pred_abs = pred_diff + cmaq_values

        for day_idx in range(day_count):
            start = day_idx * 24
            end = start + 24
            day_start_time = t0 + pd.Timedelta(hours=start)

            # strict_month_boundary=True 時，Day 2/3 若跨到下個月就不納入評估。
            if cfg.strict_month_boundary and day_start_time.month != target.month:
                continue
            day_data[day_idx].predict.extend(pred_abs[start:end])
            day_data[day_idx].obs.extend(obs_values[start:end])
            day_data[day_idx].cmaq.extend(cmaq_values[start:end])

    return day_data


def has_evaluation_data(day_data: list[DaySeries]) -> bool:
    """檢查是否至少有 Day 1 可評估資料。"""

    return bool(day_data and len(day_data[0].predict) > 0)


def write_month_outputs(
    target: MonthTarget,
    day_data: list[DaySeries],
    cfg: RuntimeConfig,
    work_dir: Path,
) -> None:
    """輸出單月評估結果。

    每個月份資料夾會輸出三類 CSV：
    1. {site}_{month}_Hourly_Check_Data_Full.csv
       用來逐筆檢查 AI 預測絕對值、Obs 與 CMAQ。
    2. {site}_{month}_Evaluation_Original_CMAQ_Full.csv
       CMAQ baseline 的 Day 1/2/3 評估。
    3. {site}_{month}_Evaluation_AI_Corrected_Full.csv
       AI 修正後的 Day 1/2/3 評估。
    """

    check_frames = []
    for idx, series in enumerate(day_data, start=1):
        # Hourly_Check_Data 是明細表，方便之後追查某個 Day 的所有小時值。
        check_frames.append(
            pd.DataFrame(
                {
                    "Day": [f"Day {idx}"] * len(series.predict),
                    f"{cfg.model_label}_Predict_Abs": series.predict,
                    "True_Obs_Abs": series.obs,
                    "CMAQ_Abs": series.cmaq,
                }
            )
        )

    check_df = pd.concat(check_frames, ignore_index=True)
    check_path = target.folder / f"{target.site}_{target.month:02d}_Hourly_Check_Data_{cfg.mode_suffix}.csv"
    check_df.to_csv(check_path, index=False)

    index_labels = [f"Day {idx}" for idx in range(1, len(day_data) + 1)]

    # 同一份 day_data 分別用 AI_Predict_Abs 與 CMAQ_Abs 對 Obs 計算指標。
    model_df = pd.DataFrame(
        [calc_metrics(series.predict, series.obs, cfg.pollutant) for series in day_data],
        columns=METRICS_COLS,
        index=index_labels,
    )
    cmaq_df = pd.DataFrame(
        [calc_metrics(series.cmaq, series.obs, cfg.pollutant) for series in day_data],
        columns=METRICS_COLS,
        index=index_labels,
    )

    cmaq_path = target.folder / f"{target.site}_{target.month:02d}_Evaluation_Original_CMAQ_{cfg.mode_suffix}.csv"
    model_path = (
        target.folder
        / f"{target.site}_{target.month:02d}_Evaluation_{cfg.model_label}_Corrected_{cfg.mode_suffix}.csv"
    )
    cmaq_df.to_csv(cmaq_path)
    model_df.to_csv(model_path)

    print(f"[eval:month] {target.folder.name}: wrote {display_path(model_path, work_dir)}")


def merge_day_data(monthly_items: Iterable[list[DaySeries]]) -> list[DaySeries]:
    """把多個月份的 Day 1/2/3 串接成年度資料。

    例如 1、4、7、10 月的 Day 1 會全部合併到年度 Day 1；
    Day 2 與 Day 3 也各自合併。這樣年度評估仍維持 Day 1/2/3 的格式。
    """

    merged: list[DaySeries] | None = None
    for day_data in monthly_items:
        if merged is None:
            merged = empty_day_series(len(day_data))
        for target_series, source_series in zip(merged, day_data):
            target_series.predict.extend(source_series.predict)
            target_series.obs.extend(source_series.obs)
            target_series.cmaq.extend(source_series.cmaq)
    return merged or []


def write_annual_outputs(
    work_dir: Path,
    group_target: MonthTarget,
    monthly_items: list[tuple[MonthTarget, list[DaySeries]]],
    cfg: RuntimeConfig,
) -> bool:
    """輸出年度評估結果。

    年度輸出會建立或使用對應的 ALL 資料夾，例如：
    TCN_ReLU-JAN-42_basecase -> TCN_ReLU-ALL-42_basecase。

    注意：呼叫這個函式前，main() 會先確認同一方案/測站已包含
    JAN、APR、JUL、OCT 四個月份，避免只選部分月份時覆蓋全年結果。
    """

    sorted_items = sorted(monthly_items, key=lambda item: item[0].month)
    day_data = merge_day_data(day_series for _, day_series in sorted_items)
    if not has_evaluation_data(day_data):
        return False

    output_folder = work_dir / group_target.annual_folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    check_frames = []
    for target, month_day_data in sorted_items:
        for idx, series in enumerate(month_day_data, start=1):
            # 年度明細多保留 Month 欄位，方便回查每筆資料來自哪個月份。
            check_frames.append(
                pd.DataFrame(
                    {
                        "Month": [target.month] * len(series.predict),
                        "Day_Type": [f"Day {idx}"] * len(series.predict),
                        f"{cfg.model_label}_Predict_Abs": series.predict,
                        "True_Obs_Abs": series.obs,
                        "CMAQ_Abs": series.cmaq,
                    }
                )
            )
    check_df = pd.concat(check_frames, ignore_index=True)
    check_path = output_folder / f"{group_target.site}_Annual_00_Hourly_Check_Data_{cfg.mode_suffix}.csv"
    check_df.to_csv(check_path, index=False)

    index_labels = [f"Day {idx}" for idx in range(1, len(day_data) + 1)]

    # 年度指標是先合併四個月份的所有小時資料，再計算一次總體指標。
    model_df = pd.DataFrame(
        [calc_metrics(series.predict, series.obs, cfg.pollutant) for series in day_data],
        columns=METRICS_COLS,
        index=index_labels,
    )
    cmaq_df = pd.DataFrame(
        [calc_metrics(series.cmaq, series.obs, cfg.pollutant) for series in day_data],
        columns=METRICS_COLS,
        index=index_labels,
    )

    cmaq_path = output_folder / f"{group_target.site}_Annual_00_Evaluation_Original_CMAQ_{cfg.mode_suffix}.csv"
    model_path = (
        output_folder
        / f"{group_target.site}_Annual_00_Evaluation_{cfg.model_label}_Corrected_{cfg.mode_suffix}.csv"
    )
    cmaq_df.to_csv(cmaq_path)
    model_df.to_csv(model_path)

    print(f"[eval:annual] {output_folder.name}: wrote {display_path(model_path, work_dir)}")
    return True


def choose_pollutant_interactively(current: str | None) -> str:
    """互動式選擇污染物。

    若命令列已經提供 --pollutant，current 不會是 None，就直接使用該值。
    否則顯示 O3/PM2.5 選單，讓使用者在程式一開始決定要跑哪一種污染物。
    """

    if current is not None:
        return current

    print("\nPollutant")
    print("  1. O3")
    print("  2. PM2.5")
    while True:
        raw = input("Choose pollutant [1/O3, 2/PM2.5] (default: 1): ").strip()
        if raw == "":
            return "o3"
        try:
            return canonical_pollutant({"1": "o3", "2": "pm2.5"}.get(raw.lower(), raw))
        except argparse.ArgumentTypeError as exc:
            print(f"[input:error] {exc}")


def default_work_dir_for_pollutant(pollutant: str, script_dir: Path) -> str:
    """依污染物決定互動模式的預設工作資料夾。

    O3 預設是本腳本所在資料夾，也就是 TCN_Transformer_o3_v2。
    PM2.5 預設切到同層的 ../TCN_Transformer_pm。
    """

    if pollutant == "pm25":
        pm_dir = script_dir.parent / "TCN_Transformer_pm"
        if pm_dir.exists():
            return "../TCN_Transformer_pm"
    return "."


def format_months(months: Iterable[int]) -> str:
    """把月份清單格式化成 1(JAN), 4(APR) 這種顯示文字。"""

    return ", ".join(f"{month}({MONTH_ABBR[month]})" for month in sorted(months))


def parse_selection(
    raw: str,
    available: set[int],
    value_name: str,
    allow_month_abbr: bool = False,
) -> list[int] | None:
    """解析互動輸入的月份或測站清單。

    支援格式：
    - 直接 Enter、all、*：代表全部。
    - 空白分隔：1 4 10 或 42 43。
    - 逗號/分號分隔：1,4,10。
    - 月份可額外輸入 JAN、APR、JUL、OCT。

    回傳 None 代表全部；回傳 list[int] 代表使用者指定的部分項目。
    """

    value = raw.strip()
    if value == "" or value.lower() in {"all", "*"}:
        return None

    selected: list[int] = []
    tokens = [token for token in re.split(r"[\s,;]+", value) if token]
    for token in tokens:
        token_upper = token.upper()
        if allow_month_abbr and token_upper in ABBR_MONTH:
            number = ABBR_MONTH[token_upper]
        else:
            try:
                number = int(token)
            except ValueError as exc:
                raise ValueError(f"{value_name} must be numbers, all, or month names.") from exc

        if number not in available:
            raise ValueError(f"{value_name} {number} is not available. Available: {sorted(available)}")
        selected.append(number)

    return sorted(set(selected))


def choose_values_interactively(
    label: str,
    available: set[int],
    current: list[int] | None,
    allow_month_abbr: bool = False,
) -> list[int] | None:
    """互動式選擇月份或測站。

    current 若已由命令列提供，就不再詢問。
    否則先顯示目前資料夾中實際掃到的可用項目，再讀取使用者輸入。
    """

    if current is not None:
        return current
    if not available:
        return None

    if allow_month_abbr:
        available_label = format_months(available)
    else:
        available_label = ", ".join(str(value) for value in sorted(available))
    print(f"\nAvailable {label}: {available_label}")

    while True:
        raw = input(f"Choose {label} (Enter/all = all): ")
        try:
            return parse_selection(raw, available, label, allow_month_abbr=allow_month_abbr)
        except ValueError as exc:
            print(f"[input:error] {exc}")


def apply_interactive_choices(args: argparse.Namespace, script_dir: Path) -> argparse.Namespace:
    """套用互動選單的所有選項。

    這裡做三件事：
    1. 選污染物。
    2. 依污染物決定預設 work_dir，並預掃可用資料夾。
    3. 讓使用者從實際可用的月份與測站中選擇要跑哪些。

    回傳同一個 args 物件，後面的 main() 可以用和命令列模式相同的流程執行。
    """

    print("Interactive selection")
    args.pollutant = choose_pollutant_interactively(args.pollutant)

    if args.work_dir is None:
        args.work_dir = default_work_dir_for_pollutant(args.pollutant, script_dir)

    work_dir = resolve_work_dir(args.work_dir, script_dir)
    if not work_dir.exists():
        raise FileNotFoundError(f"Work dir not found: {work_dir}")

    preview_targets = discover_targets(
        work_dir=work_dir,
        prediction_glob=args.prediction_glob,
        sites=None,
        months=None,
        scheme_regex=args.scheme_regex,
    )
    available_months = {target.month for target in preview_targets}
    available_sites = {target.site for target in preview_targets}

    # 先掃描一次只是為了讓使用者知道目前有哪些月份/測站可選。
    # 真正執行時 main() 會再依選擇結果掃描一次 targets。
    if preview_targets:
        print(f"[info] Work dir: {display_path(work_dir, script_dir)}")
        print(f"[info] Matching scheme folders: {len({target.folder for target in preview_targets})}")
    else:
        print(f"[warn] No prediction files found in {display_path(work_dir, script_dir)}")

    args.months = choose_values_interactively(
        label="months",
        available=available_months,
        current=args.months,
        allow_month_abbr=True,
    )
    args.sites = choose_values_interactively(
        label="sites",
        available=available_sites,
        current=args.sites,
    )

    chosen_months = "all" if args.months is None else format_months(args.months)
    chosen_sites = "all" if args.sites is None else ", ".join(str(site) for site in args.sites)
    print(f"\n[selection] Pollutant={pollutant_spec(args.pollutant).name}, months={chosen_months}, sites={chosen_sites}")
    return args


def should_run_interactive() -> bool:
    """判斷是否進入互動模式。

    沒有任何命令列參數時，預設進入互動模式。
    或者使用者明確加上 --interactive，也會進入互動模式。
    """

    return len(sys.argv) == 1 or "--interactive" in sys.argv


def parse_args() -> argparse.Namespace:
    """定義命令列參數。

    互動模式適合手動操作；命令列參數模式適合批次執行或重複測試。
    兩種模式最後都會整理成同一組 args，交給 main() 的主流程處理。
    """

    parser = argparse.ArgumentParser(
        description="Fill prediction time columns and run monthly/annual performance evaluation."
    )
    parser.add_argument("--interactive", action="store_true", help="Choose pollutant, months, and sites at startup.")
    parser.add_argument("--pollutant", type=canonical_pollutant, default=None, help="o3 or pm2.5")
    parser.add_argument("--work-dir", default=None, help="Directory that contains scheme folders.")
    parser.add_argument("--base-data", default=None, help="Path to base_data.csv, relative to --work-dir.")
    parser.add_argument("--config", default=None, help="Path to config.py, relative to --work-dir.")
    parser.add_argument("--sites", nargs="*", type=int, default=None, help="Sites to process, e.g. --sites 42 43.")
    parser.add_argument("--months", nargs="*", type=int, default=None, help="Months to process, e.g. --months 1 4.")
    parser.add_argument("--scheme-regex", default=None, help="Only process folders matching this regex.")
    parser.add_argument("--prediction-glob", default="*_ratio_avoid_predict.csv")
    parser.add_argument("--scope", choices=("time", "monthly", "annual", "both"), default="both")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--input-timesteps", type=int, default=None)
    parser.add_argument("--output-timesteps", type=int, default=None)
    parser.add_argument("--strict-month-boundary", action="store_true")
    parser.add_argument("--model-label", default="AI")
    return parser.parse_args()


def main() -> None:
    """程式進入點。

    主流程分成六個階段：
    1. 讀取命令列參數或互動式選擇。
    2. 建立 RuntimeConfig。
    3. 掃描符合月份/測站/方案條件的預測檔。
    4. 先補齊或檢查 time 欄位。
    5. 執行每月評估。
    6. 若四個月份齊全，執行年度評估。
    """

    # -------------------------------------------------------------------------
    # 階段 1：讀取使用者設定
    # -------------------------------------------------------------------------
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    # 沒有帶參數時會進入互動模式；帶參數時可直接批次執行。
    if should_run_interactive():
        args = apply_interactive_choices(args, script_dir)
    elif args.pollutant is None:
        args.pollutant = "o3"

    work_dir = resolve_work_dir(args.work_dir, script_dir)
    if not work_dir.exists():
        raise FileNotFoundError(f"Work dir not found: {work_dir}")

    # -------------------------------------------------------------------------
    # 階段 2：整合 config.py 與命令列設定
    # -------------------------------------------------------------------------
    config_defaults = load_config_defaults(work_dir, args.config)
    pollutant = pollutant_spec(args.pollutant)
    mode_suffix = "Strict" if args.strict_month_boundary else "Full"
    cfg = RuntimeConfig(
        year=args.year if args.year is not None else config_defaults["year"],
        input_timesteps=(
            args.input_timesteps
            if args.input_timesteps is not None
            else config_defaults["input_timesteps"]
        ),
        output_timesteps=(
            args.output_timesteps
            if args.output_timesteps is not None
            else config_defaults["output_timesteps"]
        ),
        strict_month_boundary=args.strict_month_boundary,
        model_label=args.model_label,
        mode_suffix=mode_suffix,
        pollutant=pollutant,
    )

    # 使用者沒有指定 sites/months 時，None 代表全部。
    sites = set(args.sites) if args.sites else None
    months = set(args.months) if args.months else None
    invalid_months = sorted(month for month in months or [] if month not in MONTH_ABBR)
    if invalid_months:
        raise ValueError(f"Unsupported month(s): {invalid_months}. Supported months: {sorted(MONTH_ABBR)}")

    # -------------------------------------------------------------------------
    # 階段 3：掃描這次要處理的預測檔
    # -------------------------------------------------------------------------
    targets = discover_targets(
        work_dir=work_dir,
        prediction_glob=args.prediction_glob,
        sites=sites,
        months=months,
        scheme_regex=args.scheme_regex,
    )
    if not targets:
        print("[done] No matching prediction files found.")
        return

    print(f"[info] Work dir: {display_path(work_dir, script_dir)}")
    print(f"[info] Pollutant: {cfg.pollutant.name}")
    print(f"[info] Targets: {len(targets)}")

    # -------------------------------------------------------------------------
    # 階段 4：時間填補
    # -------------------------------------------------------------------------
    # 評估必須依 time 去 base_data.csv 對齊 CMAQ/Obs，
    # 因此任何評估前都先確保 time 欄位正確。
    filled = 0
    for target in targets:
        if ensure_time_column(target, cfg, work_dir):
            filled += 1
    print(f"[time:done] checked={len(targets)}, filled={filled}")

    if args.scope == "time":
        return

    # -------------------------------------------------------------------------
    # 階段 5：讀取 base_data.csv 並執行每月評估
    # -------------------------------------------------------------------------
    base_data_path = find_base_data(work_dir, script_dir, args.pollutant, args.base_data)
    print(f"[info] Base data: {display_path(base_data_path, work_dir)}")

    # base_data 依測站快取；同一測站有多個月份時不用重複讀檔。
    base_cache: dict[int, pd.DataFrame] = {}

    # month_results 保留已成功處理的月份資料，後面年度評估會重複使用。
    month_results: dict[MonthTarget, list[DaySeries]] = {}
    monthly_success = 0

    for target in targets:
        if target.site not in base_cache:
            base_cache[target.site] = load_base_data(target.site, base_data_path)

        day_data = process_month_data(target, base_cache[target.site], cfg)
        if not has_evaluation_data(day_data):
            print(f"[skip] No evaluation data: {target.folder.name}")
            continue

        month_results[target] = day_data
        monthly_success += 1
        if args.scope in {"monthly", "both"}:
            write_month_outputs(target, day_data, cfg, work_dir)

    # -------------------------------------------------------------------------
    # 階段 6：年度評估
    # -------------------------------------------------------------------------
    if args.scope in {"annual", "both"}:
        grouped: dict[tuple[str, int, str], list[tuple[MonthTarget, list[DaySeries]]]] = {}
        for target, day_data in month_results.items():
            grouped.setdefault(target.group_key, []).append((target, day_data))

        annual_success = 0
        for _, items in sorted(grouped.items(), key=lambda item: item[0]):
            representative = items[0][0]
            group_months = {target.month for target, _ in items}

            # 只有四個代表月份都在同一批處理結果中，才更新 Annual 檔。
            # 這可以避免只跑 JAN 時誤把年度檔覆蓋成單月結果。
            if group_months != ANNUAL_MONTHS:
                missing = sorted(ANNUAL_MONTHS - group_months)
                print(
                    f"[skip] Annual evaluation needs all months for {representative.annual_folder_name}; "
                    f"missing={missing}"
                )
                continue
            if write_annual_outputs(work_dir, representative, items, cfg):
                annual_success += 1
        print(f"[eval:annual:done] success={annual_success}/{len(grouped)}")

    print(f"[eval:month:done] success={monthly_success}/{len(targets)}")


if __name__ == "__main__":
    main()
