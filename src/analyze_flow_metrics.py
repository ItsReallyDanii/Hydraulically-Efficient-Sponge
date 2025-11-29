import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ks_2samp

def analyze_flow_metrics(csv_path="results/flow_metrics/flow_metrics.csv"):
    # ------------------------------------------------------------------
    # 📂 Load and sanity check
    # ------------------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"🚫 Flow metrics file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]  # make case-insensitive

    required_cols = {"type", "mean_k", "mean_dp/dy", "flowrate", "porosity", "anisotropy"}
    if not required_cols.issubset(set(df.columns)):
        raise KeyError(f"Missing columns in CSV. Expected: {required_cols}, found: {set(df.columns)}")

    # ------------------------------------------------------------------
    # 🧩 Split real vs synthetic
    # ------------------------------------------------------------------
    real = df[df["type"].str.lower() == "real"]
    synth = df[df["type"].str.lower() == "synthetic"]

    if len(real) == 0 or len(synth) == 0:
        raise ValueError("❌ No real or synthetic entries found. Check the 'type' column content.")

    print(f"📊 Loaded {len(real)} real and {len(synth)} synthetic samples for comparison.\n")

    # ------------------------------------------------------------------
    # 🔬 Statistical comparison
    # ------------------------------------------------------------------
    metrics = ["mean_k", "mean_dp/dy", "flowrate", "porosity", "anisotropy"]
    results = []

    for metric in metrics:
        real_vals = real[metric].dropna().values
        synth_vals = synth[metric].dropna().values

        t_p = ttest_ind(real_vals, synth_vals, equal_var=False).pvalue
        ks_p = ks_2samp(real_vals, synth_vals).pvalue

        results.append({
            "Metric": metric,
            "real_mean": np.mean(real_vals),
            "synthetic_mean": np.mean(synth_vals),
            "t_p_value": t_p,
            "ks_p_value": ks_p
        })

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # 💾 Save results
    # ------------------------------------------------------------------
    out_path = "results/physics_validation_report.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)

    # ------------------------------------------------------------------
    # 🌿 Summary report
    # ------------------------------------------------------------------
    print("🧾 Physics validation report saved → results/physics_validation_report.csv\n")

    sig_t = (results_df["t_p_value"] > 0.05).sum()
    sig_ks = (results_df["ks_p_value"] > 0.05).sum()

    print("🌿 Flow Physics Validation Summary")
    print("-----------------------------------")
    print(f"✅ Metrics analyzed: {len(metrics)}")
    print(f"✅ Metrics statistically similar (T-test): {sig_t}/{len(metrics)}")
    print(f"✅ Metrics distribution-similar (KS-test): {sig_ks}/{len(metrics)}\n")

    print("📊 Detailed Metric Comparison:\n")
    print(results_df.to_string(index=False, float_format="%.5f"))

    print("\n🎯 Interpretation:")
    print("- p > 0.05 ⇒ statistically similar (no significant difference).")
    print("- p ≤ 0.05 ⇒ statistically different (model may need refinement).")

if __name__ == "__main__":
    analyze_flow_metrics()
