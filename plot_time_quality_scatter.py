
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    try:
        df = pd.read_csv('summary_longform.csv')
    except Exception as e:
        print(f"Error reading 'summary_longform.csv': {e}")
        exit(1)

    # Filter out rows with missing values in key columns
    key_cols = ['wall_time_baseline_s', 'avg_entropy_baseline', 'wall_time_srb_s', 'avg_entropy_srb']
    df_filtered = df.dropna(subset=key_cols)

    x_baseline = df_filtered['wall_time_baseline_s']
    y_baseline = df_filtered['avg_entropy_baseline']
    x_srb = df_filtered['wall_time_srb_s']
    y_srb = df_filtered['avg_entropy_srb']

    plt.figure(figsize=(8,6))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Scatter plots with improved point size and alpha
    plt.scatter(x_baseline, y_baseline, color='blue', marker='o', label='Baseline', s=40, alpha=0.7)
    plt.scatter(x_srb, y_srb, color='orange', marker='^', label='SRB Dynamic', s=40, alpha=0.7)

    # Trendlines
    coeffs_baseline = np.polyfit(x_baseline, y_baseline, 1)
    trendline_baseline = np.poly1d(coeffs_baseline)
    x_vals_baseline = np.linspace(x_baseline.min(), x_baseline.max(), 100)
    plt.plot(x_vals_baseline, trendline_baseline(x_vals_baseline), color='blue', linestyle='--')

    coeffs_srb = np.polyfit(x_srb, y_srb, 1)
    trendline_srb = np.poly1d(coeffs_srb)
    x_vals_srb = np.linspace(x_srb.min(), x_srb.max(), 100)
    plt.plot(x_vals_srb, trendline_srb(x_vals_srb), color='orange', linestyle='--')

    plt.xlabel('Wall Time (seconds)')
    plt.ylabel('Average Entropy')
    plt.title('Wall Time vs. Semantic Entropy – SRB vs Baseline')
    plt.legend()

    avg_diff_wall_time = (x_baseline - x_srb).mean()
    avg_diff_entropy = (y_srb - y_baseline).mean()

    print(f"Average difference in wall time (baseline - SRB): {avg_diff_wall_time:.4f} seconds")
    print(f"Average difference in entropy (SRB - baseline): {avg_diff_entropy:.4f}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('time_quality_scatter.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
