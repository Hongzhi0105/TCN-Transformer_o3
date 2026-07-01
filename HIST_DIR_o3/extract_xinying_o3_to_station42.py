from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path("F:/")
STATION_NAME = "新營"
STATION_ID = 42
ITEM_NAME = "O3"
ROC_YEARS = (102, 103, 104)
TEMPLATE_PATH = ROOT / "HIST_DIR_pm" / f"station{STATION_ID}.xlsx"
OUTPUT_PATH = ROOT / "HIST_DIR_o3" / f"station{STATION_ID}.xlsx"


def add_local_dependency_path(root: Path) -> None:
    """Use the local xlrd install created for reading old .xls files, if present."""
    local_deps = root / "python_excel_deps"
    if local_deps.exists():
        sys.path.insert(0, str(local_deps))


def find_station_file(root: Path, roc_year: int) -> Path:
    folder = root / f"{roc_year}_HOUR_00"
    if not folder.exists():
        raise FileNotFoundError(f"找不到資料夾：{folder}")

    pattern = f"{roc_year}年{STATION_NAME}站_*.xls"
    matches = sorted(
        p for p in folder.rglob(pattern)
        if p.is_file() and not any(part.lower() == "outputs" for part in p.parts)
    )
    if not matches:
        raise FileNotFoundError(f"在 {folder} 找不到符合 {pattern} 的檔案")
    if len(matches) > 1:
        print(f"警告：{folder} 找到多個檔案，使用第一個：{matches[0]}")
    return matches[0]


def read_o3_hourly_values(xls_path: Path) -> dict[tuple[int, int, int], float]:
    df = pd.read_excel(xls_path, sheet_name=0, dtype=object)
    df.columns = [str(col).strip() for col in df.columns]

    required = {"日期", "測站", "測項"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{xls_path} 缺少必要欄位：{sorted(missing)}")

    hour_cols = []
    for col in df.columns:
        label = str(col).strip()
        if label.isdigit() and 0 <= int(label) <= 23:
            hour_cols.append((col, int(label)))
    if len(hour_cols) != 24:
        raise ValueError(f"{xls_path} 小時欄位不是 24 個，目前找到 {len(hour_cols)} 個")

    rows = df[df["測項"].astype(str).str.strip().eq(ITEM_NAME)].copy()
    if rows.empty:
        raise ValueError(f"{xls_path} 找不到測項 {ITEM_NAME}")

    values: dict[tuple[int, int, int], float] = {}
    for _, row in rows.iterrows():
        date = pd.to_datetime(row["日期"], errors="coerce")
        if pd.isna(date):
            continue
        for col, hour in hour_cols:
            key = (int(date.month), int(date.day), hour)
            value = pd.to_numeric(row[col], errors="coerce")
            values[key] = None if pd.isna(value) else float(value)

    return values


def load_time_template(template_path: Path) -> pd.DataFrame:
    if template_path.exists():
        template = pd.read_excel(template_path, usecols=["time"])
        template["time"] = pd.to_datetime(template["time"])
        return template

    time_index = pd.date_range("2016-01-01 00:00:00", "2016-12-31 23:00:00", freq="h")
    return pd.DataFrame({"time": time_index})


def build_output(root: Path, output_path: Path) -> pd.DataFrame:
    output = load_time_template(TEMPLATE_PATH if root == ROOT else root / "HIST_DIR_pm" / f"station{STATION_ID}.xlsx")

    for roc_year in ROC_YEARS:
        gregorian_year = roc_year + 1911
        source = find_station_file(root, roc_year)
        hourly = read_o3_hourly_values(source)

        output[str(gregorian_year)] = [
            hourly.get((ts.month, ts.day, ts.hour)) for ts in output["time"]
        ]
        print(f"{gregorian_year}: {source} -> {len(hourly)} hourly values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_excel(output_path, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Xinying station O3 data from 102-104 HOUR_00 folders and merge like HIST_DIR_pm/station42.xlsx."
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="資料根目錄，預設 F:/")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="輸出 xlsx 路徑")
    args = parser.parse_args()

    add_local_dependency_path(args.root)
    result = build_output(args.root, args.output)
    print(f"完成輸出：{args.output}")
    print(f"資料筆數：{len(result)} rows x {len(result.columns)} columns")


if __name__ == "__main__":
    main()
