# region 0.00 Project Overview
# This script estimates a static Taylor rule for Poland using quarterly macroeconomic data.
#
# MODEL SPECIFICATION:
#   i_t = c + alpha * (pi_t - pi_target_t) + beta * output_gap_t + epsilon_t
#
# where:
# - i_t is the NBP policy rate
# - pi_t is quarterly inflation
# - pi_target_t is the inflation target
# - output_gap_t is the quarterly output gap
#
# METHODOLOGICAL NOTES:
# - The model uses a static Taylor rule without lagged interest rates.
# - Inflation is computed from monthly HICP data and aggregated to quarterly frequency. (https://data.ecb.europa.eu/data/datasets/ICP/ICP.M.PL.N.000000.4.ANR)
# - The output gap is estimated from quarterly real GDP using an HP filter. (https://fred.stlouisfed.org/series/NGDPRSAXDCPLQ)
#       - !Run the output-gap-builder.py script to generate the required Excel file for the output gap!
# - Policy rates are converted from RRP decision dates to quarterly average rates. (https://nbp.pl/en/historic-interest-rates/)
# - A time-varying inflation target is used for the early inflation-targeting period in Poland.
#
# OUTPUTS:
# - Estimated Taylor rule coefficients
# - Implied equilibrium policy rate
# - Charts and summary tables
# - Comparison across two samples:
#     1. Full sample (1998-2025)
#     2. Post-2004 sample (2004-2025, after constant 2.5% target)
#
# MAIN REFERENCES:
# - Taylor (1993): Discretion versus policy rules in practice
# - Standard HP filter approach for cyclical decomposition of real GDP
# endregion

#%% 
# region 1.1 Imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
from visualization import plot_taylor_rule_timeseries, plot_policy_stance
# endregion


#%% 
# region 1.2 Constants
INFLATION_TARGET = 2.5
MIN_OBS_REGRESSION = 20
DATA_PATH = "data/"
# endregion


#%% 
# region 2.0 Data loaders

def load_pl_policy_rate_quarterly(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Worksheet")

    df = df.rename(columns={
        "Data": "Date",
        "Stopa referencyjna": "Policy_Rate"
    })

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Policy_Rate"] = pd.to_numeric(df["Policy_Rate"], errors="coerce")

    df = df.dropna().sort_values("Date")
    df = df.set_index("Date")

    daily = df["Policy_Rate"].resample("D").ffill()

    return (
        daily.resample("QE")
        .mean()
        .to_frame("Policy_Rate")
        .reset_index()
    )


def load_pl_hicp_quarterly_yoy(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Worksheet", usecols=[0, 1])
    df.columns = ["Date", "HICP"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["HICP"] = pd.to_numeric(df["HICP"], errors="coerce")

    df = df.dropna().sort_values("Date")

    df["Inflation"] = (df["HICP"] / df["HICP"].shift(12) - 1) * 100

    return (
        df.dropna(subset=["Inflation"])
          .set_index("Date")[["Inflation"]]
          .resample("QE")
          .mean()
          .reset_index()
    )


def load_pl_output_gap_quarterly(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Worksheet")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Output_Gap"] = pd.to_numeric(df["Output_Gap"], errors="coerce")

    df = df.dropna().sort_values("Date")

    # align to quarter end
    df["Date"] = df["Date"] + pd.offsets.QuarterEnd(0)

    return df[["Date", "Output_Gap"]]

# endregion


#%% 
# region 3.0 Build dataset

def get_inflation_target(date):
    y = date.year
    if y <= 1999:
        return 7.2
    elif y == 2000:
        return 6.1
    elif y == 2001:
        return 6.0
    elif y == 2002:
        return 3.0
    else:
        return 2.5


def build_master_dataframe():
    infl = load_pl_hicp_quarterly_yoy(f"{DATA_PATH}/PL-CPI.xlsx")
    gap = load_pl_output_gap_quarterly(f"{DATA_PATH}/PL-OutputGap.xlsx")
    rate = load_pl_policy_rate_quarterly(f"{DATA_PATH}/PL-BaseRate.xlsx")

    df = (
        infl.merge(gap, on="Date", how="inner")
            .merge(rate, on="Date", how="inner")
            .sort_values("Date")
            .reset_index(drop=True)
    )

    df["Inflation_Target"] = df["Date"].apply(get_inflation_target)
    df["Inflation_Gap"] = df["Inflation"] - df["Inflation_Target"]

    return df

# endregion


#%% 
# region 4.0 OLS

def run_ols(y, X):
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit(cov_type="HC1")

# endregion


#%% 
# region 4.1 Taylor estimation

def estimate_taylor_rule_static_pl(df, start_date=None):

    df = df.dropna(subset=["Inflation_Gap", "Output_Gap", "Policy_Rate"])

    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]

    if len(df) < MIN_OBS_REGRESSION:
        return None

    model = run_ols(
        df["Policy_Rate"],
        df[["Inflation_Gap", "Output_Gap"]]
    )

    latest = df.iloc[-1]

    c = model.params["const"]
    a = model.params["Inflation_Gap"]
    b = model.params["Output_Gap"]

    eq = c + a * latest["Inflation_Gap"] + b * latest["Output_Gap"]

    return {
        "policy_rate": latest["Policy_Rate"],
        "inflation": latest["Inflation"],
        "inflation_gap": latest["Inflation_Gap"],
        "output_gap": latest["Output_Gap"],
        "equilibrium_rate_regression": eq,
        "const_estimated": c,
        "alpha_estimated": a,
        "beta_estimated": b,
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs)
    }

# endregion


#%% 
# region 5.0 MAIN

if __name__ == "__main__":
    import os
    os.makedirs("result", exist_ok=True)

    print("=" * 80)
    print("POLAND TAYLOR RULE ANALYSIS")
    print("=" * 80)
    print("This script:")
    print("1. loads inflation, output gap, and policy rate data")
    print("2. builds a quarterly dataset for Poland")
    print("3. estimates a static Taylor rule")
    print("4. compares full-sample and post-2004 results")
    print("5. saves summary tables and charts")
    print()

    # --- Build dataset ---
    print("[1/4] Building quarterly master dataset...")
    pl_df = build_master_dataframe()

    print(f"Dataset ready: {len(pl_df)} quarterly observations")
    print(f"Sample period: {pl_df['Date'].min().strftime('%Y-%m-%d')} to {pl_df['Date'].max().strftime('%Y-%m-%d')}")
    print("Variables available:")
    print(" - Inflation")
    print(" - Output_Gap")
    print(" - Policy_Rate")
    print(" - Inflation_Gap")
    print()

    # --- Run models ---
    print("[2/4] Estimating Taylor rule models...")
    results = {
        "Full sample": estimate_taylor_rule_static_pl(pl_df),
        "Post-2004": estimate_taylor_rule_static_pl(pl_df, "2004-01-01")
    }
    print()

    # --- Print summary ---
    print("[3/4] Interpreting model results...")
    rows = []

    for name, r in results.items():
        if r is None:
            print(f"{name}: insufficient data")
            continue

        diff = r["policy_rate"] - r["equilibrium_rate_regression"]
        stance = "restrictive" if diff > 0 else "accommodative"

        print("-" * 80)
        print(name.upper())
        print("-" * 80)
        print(f"Current policy rate:        {r['policy_rate']:.2f}%")
        print(f"Model-implied Taylor rate:  {r['equilibrium_rate_regression']:.2f}%")
        print(f"Policy stance:              {diff:+.2f} pp ({stance})")
        print()
        print("Latest macro inputs:")
        print(f" - Inflation:               {r['inflation']:.2f}%")
        print(f" - Inflation gap:           {r['inflation_gap']:.2f} pp")
        print(f" - Output gap:              {r['output_gap']:.2f} pp")
        print()
        print("Estimated coefficients:")
        print(f" - Constant:                {r['const_estimated']:.3f}")
        print(f" - Alpha (inflation gap):   {r['alpha_estimated']:.3f}")
        print(f" - Beta (output gap):       {r['beta_estimated']:.3f}")
        print(f" - R-squared:               {r['r_squared']:.3f}")
        print(f" - Observations:            {r['n_obs']}")
        print()

        rows.append({**{"Run": name}, **r})

    # --- Save CSV ---
    print("[4/4] Saving outputs...")
    df_out = pd.DataFrame(rows).round(4)

    summary_path = "result/taylor_summary.csv"
    df_out.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")

    # --- Charts ---
    chart_1 = "result/taylor_timeseries.png"
    plot_taylor_rule_timeseries(pl_df, results, chart_1)
    print(f"Saved chart:         {chart_1}")

    if results["Full sample"] is not None:
        chart_2 = "result/policy_stance.png"
        plot_policy_stance(pl_df, results["Full sample"], chart_2)
        print(f"Saved chart:         {chart_2}")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Check the /result folder for exported outputs. & short the VIX! \n")

# endregion