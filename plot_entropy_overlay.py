import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
import os

# Expected schema for runs_longform.csv based on the user's sample
# Columns: backend, model_id, bucket, name, prompt_ix, prompt, strategy,
#          tokens_emitted, wall_time_s, avg_entropy, final_entropy, stop_reason, response_text

STRATEGY_BASELINE = "static"
STRATEGY_SRB = "srb_dynamic"
REQUIRED_COLS = {"prompt", "strategy", "avg_entropy"}


def load_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found in the current directory: {os.getcwd()}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{path}' is empty or corrupted.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading '{path}': {e}")
        return None


def have_cols(df: Optional[pd.DataFrame], cols: set) -> bool:
    return df is not None and cols.issubset(set(df.columns))


def slugify(text: str) -> str:
    # Simple filesystem-friendly slug
    return (
        str(text).lower()
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace("|", "-")
        .replace(":", "-")
        .replace("'", "")
        .replace('"', "")
    )[:64]


def make_strategy_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by prompt with columns 'avg_entropy_static' and 'avg_entropy_srb_dynamic'.
    If multiple runs exist per (prompt, strategy), they are averaged.
    """
    pivot = (
        df.groupby(["prompt", "strategy"], as_index=False)["avg_entropy"].mean()
          .pivot(index="prompt", columns="strategy", values="avg_entropy")
    )
    # Normalize friendly column names for consistent downstream access
    pivot = pivot.rename(columns={
        STRATEGY_BASELINE: "avg_entropy_static",
        STRATEGY_SRB: "avg_entropy_srb_dynamic",
    })
    return pivot


def top_prompts_by_entropy_gap(df: pd.DataFrame, topk: int = 8) -> list[str]:
    pivot = make_strategy_pivot(df)
    # Keep only prompts where we have BOTH strategies
    pivot = pivot.dropna(subset=["avg_entropy_static", "avg_entropy_srb_dynamic"], how="any")
    if pivot.empty:
        return []
    pivot["abs_gap"] = (pivot["avg_entropy_static"] - pivot["avg_entropy_srb_dynamic"]).abs()
    top = pivot.sort_values("abs_gap", ascending=False).head(topk)
    return list(top.index)


def plot_avg_entropy_bars(df: pd.DataFrame, prompts: list[str]):
    if not prompts:
        print("No prompts to plot.")
        return
    pivot = make_strategy_pivot(df)

    for prompt in prompts:
        if prompt not in pivot.index:
            continue
        row = pivot.loc[prompt]
        if pd.isna(row.get("avg_entropy_static")) or pd.isna(row.get("avg_entropy_srb_dynamic")):
            # Need both strategies to compare visually
            continue

        base = float(row["avg_entropy_static"])  # type: ignore[index]
        srb = float(row["avg_entropy_srb_dynamic"])  # type: ignore[index]

        plt.figure(figsize=(6, 4))
        plt.bar(["Baseline", "SRB"], [base, srb])
        plt.title(f"Average Entropy – {prompt}", fontsize=10, wrap=True, pad=20)
        plt.ylabel("Average Entropy")
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.subplots_adjust(top=0.8)

        filename = f"entropy_avg_{slugify(prompt)}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved average-entropy bar chart for prompt '{prompt}' as '{filename}'")


def main():
    # We now rely solely on runs_longform.csv (per-run aggregates). No token_step data is required.
    runs_df = load_csv("runs_longform.csv")

    if not have_cols(runs_df, REQUIRED_COLS):
        print(
            "Error: 'runs_longform.csv' must contain at least these columns: "
            f"{sorted(REQUIRED_COLS)}. Found: {list(runs_df.columns) if runs_df is not None else 'None'}"
        )
        return

    # Choose top prompts by absolute average-entropy gap
    prompts = top_prompts_by_entropy_gap(runs_df, topk=8)
    if not prompts:
        print("Warning: Could not identify prompts that contain BOTH strategies ('static' and 'srb_dynamic').\n"
              "Ensure your CSV has rows for each prompt with both strategies present.")
    plot_avg_entropy_bars(runs_df, prompts)


if __name__ == "__main__":
    main()
