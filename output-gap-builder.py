# This code read quarterly real GDP for Poland from CSV, computes HP-filter output gap
# and export an Excel file that can be used by the main Taylor rule program.

# Data source: https://fred.stlouisfed.org/series/NGDPRSAXDCPLQ

# Expected input columns:
#   - observation_date
#   - NGDPRSAXDCPLQ

# Output Excel:
#   - sheet name: Worksheet
#   - columns:
#       - Date
#       - Output_Gap

#   Output_Gap is computed as:
#    - HP cycle of log(real GDP) * 100 which approximates % deviation from trend.
    
#%%
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
import matplotlib.pyplot as plt

#%%
def build_output_gap_from_real_gdp():

    input_csv = Path("data/PL-RealGDP.csv")
    output_xlsx = Path("data/PL-OutputGap.xlsx")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    # 1. Load data
    df = pd.read_csv(input_csv, sep=None, engine="python")

    # 2. Standardize column names
    df = df.rename(columns={
        "observation_date": "Date",
        "NGDPRSAXDCPLQ": "Real_GDP"
    })

    required_cols = {"Date", "Real_GDP"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns. Found columns: {list(df.columns)}")

    # 3. Convert data types
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Real_GDP"] = pd.to_numeric(df["Real_GDP"], errors="coerce")

    # 4. Clean and sort
    df = df.dropna(subset=["Date", "Real_GDP"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    if (df["Real_GDP"] <= 0).any():
        raise ValueError("Real_GDP contains non-positive values; log cannot be computed.")

    # 5. Take logs
    df["log_GDP"] = np.log(df["Real_GDP"])

    # 6. Apply HP filter (lambda=1600 for quarterly data)
    cycle, trend = hpfilter(df["log_GDP"], lamb=1600)

    # 7. Compute output gap (approx. % deviation from trend)
    df["Output_Gap"] = cycle * 100

    # 8. Save to Excel
    export_df = df[["Date", "Output_Gap"]].copy()
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_excel(output_xlsx, index=False, sheet_name="Worksheet")

    # 9. Visualization
    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"], df["Output_Gap"], linewidth=2)
    plt.axhline(0, linestyle="--")
    plt.title("Poland Output Gap (HP filter)")
    plt.ylabel("Output gap (%)")
    plt.grid(alpha=0.3)
    plt.show()

    print(f"Sample mean (should be close to 0): {df['Output_Gap'].mean():.12f}")
    print(f"Output gap saved to: {output_xlsx}")


if __name__ == "__main__":
    build_output_gap_from_real_gdp()

