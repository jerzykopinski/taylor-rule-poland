import matplotlib.pyplot as plt
import numpy as np

def plot_taylor_rule_timeseries(df, results_dict, output_path=None):
    plt.figure(figsize=(10, 5))

    plt.plot(df["Date"], df["Policy_Rate"], label="Policy Rate")

    for name, r in results_dict.items():
        if r is None:
            continue
        
        c = r["const_estimated"]
        a = r["alpha_estimated"]
        b = r["beta_estimated"]

        fitted = c + a * df["Inflation_Gap"] + b * df["Output_Gap"]
        plt.plot(df["Date"], fitted, linestyle="--", label=f"Taylor ({name})")

    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.title("Poland: Taylor Rule vs Policy Rate")
    plt.ylabel("Rate (%)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
        plt.close()
    else:
        plt.show()


def plot_policy_stance(df, results, output_path=None):
    c = results["const_estimated"]
    a = results["alpha_estimated"]
    b = results["beta_estimated"]

    fitted = c + a * df["Inflation_Gap"] + b * df["Output_Gap"]
    gap = df["Policy_Rate"] - fitted

    plt.figure(figsize=(10, 4))

    plt.plot(df["Date"], gap, label="Policy stance")
    plt.axhline(0, linestyle="--")

    plt.title("Policy Stance (Actual - Taylor)")
    plt.ylabel("pp")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
        plt.close()
    else:
        plt.show()