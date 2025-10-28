import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # --- Load data ---
    df = pd.read_csv("summary_longform.csv")

    # --- Derive missing metrics from the CSV columns present ---
    # The CSV provides: tokens_baseline, tokens_srb, wall_time_baseline_s, wall_time_srb_s, resonance_efficiency
    if not {"tokens_change_pct", "wall_time_change_pct"}.issubset(df.columns):
        df["tokens_change_pct"] = (df["tokens_srb"] - df["tokens_baseline"]) / df["tokens_baseline"] * 100.0
        df["wall_time_change_pct"] = (df["wall_time_srb_s"] - df["wall_time_baseline_s"]) / df["wall_time_baseline_s"] * 100.0

    # --- Aggregate by bucket ---
    agg_df = (
        df.groupby("bucket", as_index=False)
          .agg({
              "tokens_change_pct": "mean",
              "wall_time_change_pct": "mean",
              "resonance_efficiency": "mean",
          })
          .sort_values("bucket")
          .reset_index(drop=True)
    )

    # --- Build the matrix for the heatmap (metrics x buckets) ---
    metrics = ["tokens_change_pct", "wall_time_change_pct", "resonance_efficiency"]
    pretty_labels = ["Tokens Δ (%)", "Wall Time Δ (%)", "Mean Resonance Eff."]

    heatmap_data = agg_df.set_index("bucket")[metrics].T

    # --- Z-score scaling along each metric row (for color normalization) ---
    heatmap_z = heatmap_data.apply(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else 1.0), axis=1)

    # --- Text annotations: show % for deltas, 3 decimals for efficiency ---
    annotations = np.vstack([
        [f"{v:.1f}%" for v in agg_df["tokens_change_pct"]],
        [f"{v:.1f}%" for v in agg_df["wall_time_change_pct"]],
        [f"{v:.3f}" for v in agg_df["resonance_efficiency"]],
    ])

    # --- Plot ---
    plt.figure(figsize=(max(8, 1.2 * len(agg_df)), 3.6))
    ax = sns.heatmap(
        heatmap_z,
        annot=annotations,
        fmt="",
        cmap="coolwarm",
        cbar_kws={"label": "Z-score (within metric)"},
        linewidths=0.5,
        linecolor="white",
        square=True
    )

    # Axis labels/ticks
    ax.set_yticklabels(pretty_labels, rotation=0, fontsize=10)
    ax.set_xticklabels(agg_df["bucket"], rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Metric")

    # Title
    plt.title(
        "Resonance Efficiency Heatmap by Bucket\n(SRB vs Baseline — Δ% and Mean Resonance Efficiency)",
        fontsize=14,
        fontweight="bold",
        pad=16
    )

    # Caption (wrapped)
    caption = (
        "Cells show z-scored values for visual contrast; annotations display actual metrics. "
        "Tokens Δ (%) = 100*(tokens_srb - tokens_baseline)/tokens_baseline; "
        "Wall Time Δ (%) = 100*(time_srb - time_baseline)/time_baseline. "
        "Mean Resonance Eff. is the average of per-prompt resonance_efficiency within each bucket. "
        "Negative Δ indicates SRB reduced usage/time versus baseline."
    )
    plt.gcf().text(0.5, -0.12, caption, ha="center", va="top", fontsize=9, wrap=True)

    plt.tight_layout()
    plt.savefig("resonance_efficiency_heatmap_captioned.png", dpi=300, bbox_inches="tight")
    plt.savefig("resonance_efficiency_heatmap_captioned.pdf", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
