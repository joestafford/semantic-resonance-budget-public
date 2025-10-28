import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def safe_pct_change(num: pd.Series, den: pd.Series) -> pd.Series:
    """Percent change with robust divide-by-zero handling."""
    num = num.astype(float)
    den = den.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = (num - den) / den * 100.0
    # Replace inf/-inf with NaN then fill with 0 (no baseline to compare)
    pct = pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pct

def build_grouped(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby('bucket').agg({
        'tokens_baseline': 'mean',
        'tokens_srb': 'mean',
        'wall_time_baseline_s': 'mean',
        'wall_time_srb_s': 'mean',
        'resonance_efficiency': 'mean',
        'name': 'count'
    }).rename(columns={'name': 'num_prompts'})

    grouped['tokens_pct_change'] = safe_pct_change(grouped['tokens_srb'], grouped['tokens_baseline'])
    grouped['wall_time_pct_change'] = safe_pct_change(grouped['wall_time_srb_s'], grouped['wall_time_baseline_s'])
    return grouped

def prepare_heatmap_frames(grouped: pd.DataFrame):
    # Data for coloring (z-scored per column so different units are comparable)
    heatmap_cols = ['tokens_pct_change', 'wall_time_pct_change', 'resonance_efficiency']
    color_df = grouped[heatmap_cols].copy()
    color_df = (color_df - color_df.mean(axis=0)) / color_df.std(axis=0).replace(0, 1)

    # Data for annotation (human-readable)
    display_df = grouped[heatmap_cols].copy()
    display_df = display_df.rename(columns={
        'tokens_pct_change': 'Tokens % Change',
        'wall_time_pct_change': 'Wall Time % Change',
        'resonance_efficiency': 'Resonance Efficiency'
    })

    color_df.columns = display_df.columns
    return color_df, display_df

def draw_heatmap(color_df: pd.DataFrame, display_df: pd.DataFrame, grouped: pd.DataFrame, outfile: Path):
    plt.figure(figsize=(10, 2 + 0.9 * len(color_df)))
    ax = sns.heatmap(
        color_df,
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        center=0,
        annot=False,
        cbar_kws={'label': 'Column-wise z-score'}
    )

    # Annotate with readable values
    for y in range(display_df.shape[0]):
        for x in range(display_df.shape[1]):
            col = display_df.columns[x]
            val = display_df.iloc[y, x]
            if col.endswith('% Change'):
                sgn = '+' if val >= 0 else ''
                text = f"{sgn}{val:.0f}%"
            else:
                text = f"{val:.2f}"
            ax.text(x + 0.5, y + 0.5, text, ha='center', va='center', color='black', fontsize=11)

    ax.set_yticklabels(display_df.index, rotation=0, fontsize=11)
    ax.set_xticklabels(display_df.columns, rotation=20, ha='right', fontsize=11)
    ax.set_title("Resonance Efficiency by Category (SRB vs Baseline)\n"
                 "Values annotated; colors show column-wise z-score for comparability.",
                 pad=16, fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

def summarize(grouped: pd.DataFrame):
    best_bucket = grouped['resonance_efficiency'].idxmax()
    best_val = grouped.loc[best_bucket, 'resonance_efficiency']

    # Improvement proxies (negative = better if we care about reduction)
    best_tokens_red = grouped['tokens_pct_change'].idxmin()
    best_tokens_val = grouped.loc[best_tokens_red, 'tokens_pct_change']

    best_time_red = grouped['wall_time_pct_change'].idxmin()
    best_time_val = grouped.loc[best_time_red, 'wall_time_pct_change']

    print(f"Bucket with highest resonance efficiency: {best_bucket} ({best_val:.2f})")
    print(f"Largest token reduction: {best_tokens_red} ({best_tokens_val:.0f}%)")
    print(f"Largest wall-time reduction: {best_time_red} ({best_time_val:.0f}%)")

def main():
    parser = argparse.ArgumentParser(description="Plot resonance efficiency heatmap by category.")
    parser.add_argument('--input', default='summary_longform.csv', help='Path to summary_longform.csv')
    parser.add_argument('--output', default='resonance_efficiency_heatmap.png', help='Output image path')
    parser.add_argument('--sort', default='resonance_efficiency',
                        choices=['resonance_efficiency', 'tokens_pct_change', 'wall_time_pct_change', 'bucket'],
                        help='Sort rows by this column (desc for efficiency, asc for deltas).')
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"Error: File '{csv_path}' not found.")
        return

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{csv_path}' is empty.")
        return
    except Exception as e:
        print(f"Error reading '{csv_path}': {e}")
        return

    required = {'bucket', 'tokens_baseline', 'tokens_srb',
                'wall_time_baseline_s', 'wall_time_srb_s', 'resonance_efficiency'}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: Missing required columns: {sorted(missing)}")
        return

    grouped = build_grouped(df)

    # Optional sorting
    if args.sort == 'bucket':
        grouped = grouped.sort_index()
    elif args.sort in grouped.columns:
        asc = args.sort != 'resonance_efficiency'
        grouped = grouped.sort_values(by=args.sort, ascending=asc)

    # Save a CSV summary too
    summary_out = Path('bucket_summary.csv')
    grouped.round(3).to_csv(summary_out)
    print(f"Wrote bucket summary to: {summary_out.resolve()}")

    color_df, display_df = prepare_heatmap_frames(grouped)
    outfile = Path(args.output)
    draw_heatmap(color_df, display_df, grouped, outfile)
    print(f"Saved heatmap to: {outfile.resolve()}")

    summarize(grouped)

if __name__ == "__main__":
    main()
