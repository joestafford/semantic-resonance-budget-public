#!/usr/bin/env python3
"""
Format runs_longform.csv into an easy-to-read Markdown report.

- Pairs baseline ("static") and SRB ("srb_dynamic") runs by (bucket, name, prompt_ix).
- Emits a single Markdown file with:
  * Overall summary counts
  * A compact table for each prompt showing baseline vs SRB metrics
  * <details> blocks containing the full responses for both runs
- Optionally also writes a machine-friendly paired CSV for further analysis.

Usage:
  python format_runs_longform.py \
      --input runs_longform.csv \
      --output runs_longform_report.md \
      --paired-csv runs_longform_paired.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from textwrap import indent, fill
import html

import pandas as pd
from typing import Optional


def parse_args():
    p = argparse.ArgumentParser(description="Format longform runs into Markdown/HTML")
    p.add_argument("--input", required=True, help="Path to runs_longform.csv")
    p.add_argument("--output", default="runs_longform_report.md", help="Output Markdown file")
    p.add_argument("--paired-csv", default=None, help="Optional: write paired metrics to CSV")
    p.add_argument("--wrap-width", type=int, default=90, help="Soft wrap width for prompt text")
    p.add_argument("--truncate-response", type=int, default=None,
                   help="If set, truncate response_text to N characters inside details blocks")
    p.add_argument("--html-output", default=None,
                   help="If set, also write a standalone HTML report to this path (e.g., runs_longform_report.html)")
    return p.parse_args()


REQUIRED_COLS = [
    "backend","model_id","bucket","name","prompt_ix","prompt","strategy",
    "tokens_emitted","wall_time_s","avg_entropy","final_entropy",
    "stop_reason","response_text"
]

def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: Missing required columns: {missing}\n"
                         f"Columns found: {list(df.columns)}")


def safe_truncate(text: str, n: Optional[int]) -> str:
    if n is None or n <= 0:
        return text
    if len(text) <= n:
        return text
    return text[:n] + "…"


def format_float(x, digits=3):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"


def pct_change(new, old, digits=1):
    try:
        new_f, old_f = float(new), float(old)
        if old_f == 0:
            return "—"
        return f"{((new_f - old_f) / old_f) * 100:.{digits}f}%"
    except Exception:
        return "—"


def build_html_report(merged: pd.DataFrame, source_name: str, wrap_width: int, truncate_response: Optional[int]) -> str:
    # Basic CSS for readability and print friendliness
    css = """
    <style>
      :root { --fg:#1f2328; --muted:#57606a; --bg:#ffffff; --accent:#0969da; --ok:#1a7f37; --warn:#9a6700; }
      body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,Apple Color Emoji,Segoe UI Emoji; color: var(--fg); background: var(--bg); margin: 40px; }
      h1,h2,h3 { line-height: 1.25; }
      h1 { margin-top: 0; }
      .muted { color: var(--muted); }
      .bucket { margin-top: 2.0rem; }
      .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 0 rgba(27,31,36,0.04); }
      .prompt { background: #f6f8fa; border-left: 4px solid #d0d7de; padding: 12px 16px; white-space: pre-wrap; }
      table { width: 100%; border-collapse: collapse; margin-top: 8px; }
      th, td { border-bottom: 1px solid #d8dee4; padding: 8px; text-align: right; }
      th:first-child, td:first-child { text-align: left; }
      details { margin-top: 8px; }
      summary { cursor: pointer; font-weight: 600; }
      pre { background: #0b0d0e; color: #e6edf3; padding: 12px; border-radius: 6px; overflow: auto; }
      .delta-pos { color: var(--ok); font-weight: 600; }
      .delta-neg { color: var(--warn); font-weight: 600; }
      .toc { margin: 12px 0 20px 0; padding: 12px; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; }
      .toc a { text-decoration: none; color: var(--accent); }
      .pill { display: inline-block; padding: 2px 8px; border:1px solid #d0d7de; border-radius: 999px; margin-right: 6px; background:#fff; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 8px; }
      .footer { margin-top: 28px; font-size: 0.9em; color: var(--muted); }
      code.k { background:#f6f8fa; padding: 0 6px; border:1px solid #d0d7de; border-radius: 6px; }
    </style>
    """

    def f(x, d=3):
        try:
            return f"{float(x):.{d}f}"
        except Exception:
            return "-"

    def pct(new, old, digits=1):
        try:
            new_f, old_f = float(new), float(old)
            if old_f == 0:
                return "—"
            val = ((new_f - old_f) / old_f) * 100.0
            cls = "delta-pos" if val <= 0 else "delta-neg"  # lower is green for tokens/time
            return f"<span class='{cls}'>{val:.{digits}f}%</span>"
        except Exception:
            return "—"

    def esc(s: str) -> str:
        return html.escape("" if s is None or pd.isna(s) else str(s))

    # Build a simple TOC (by buckets)
    buckets = merged["bucket"].fillna("unknown").unique().tolist()
    toc_links = " · ".join(f"<a href='#{b}'>{esc(b)}</a>" for b in buckets)

    lines = []
    lines.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8' />")
    lines.append(f"<title>SRB Longform Report – {esc(source_name)}</title>")
    lines.append("<meta name='viewport' content='width=device-width, initial-scale=1' />")
    lines.append(css)
    lines.append("</head><body>")
    lines.append(f"<h1>SRB Longform Report</h1>")
    lines.append(f"<div class='muted'>Source: <code class='k'>{esc(source_name)}</code> · Total prompts paired: {merged.shape[0]}</div>")
    lines.append("<div class='toc'><strong>Buckets:</strong> " + toc_links + "</div>")

    # Overview pills by bucket count
    counts = merged["bucket"].value_counts().to_dict()
    if counts:
        lines.append("<div class='grid'>")
        for b, n in counts.items():
            lines.append(f"<div class='pill'>{esc(b)}: {n} prompts</div>")
        lines.append("</div>")

    # Sections per bucket
    last_bucket = None
    for _, row in merged.iterrows():
        bucket = esc(row.get("bucket", "unknown"))
        name   = esc(row.get("name", "unknown"))
        pidx   = row.get("prompt_ix", "-")
        prompt = row.get("prompt", "")

        if bucket != last_bucket:
            lines.append(f"<h2 id='{bucket}' class='bucket'>{bucket}</h2>")
            last_bucket = bucket

        lines.append("<div class='card'>")
        lines.append(f"<h3><code class='k'>{name}</code> <span class='muted'>(idx: {pidx})</span></h3>")
        if isinstance(prompt, str) and prompt.strip():
            lines.append(f"<div class='prompt'>{esc(prompt)}</div>")

        # Metrics table
        bt = row.get("base_tokens_emitted")
        st = row.get("srb_tokens_emitted")
        bw = row.get("base_wall_time_s")
        sw = row.get("srb_wall_time_s")
        be = row.get("base_avg_entropy")
        se = row.get("srb_avg_entropy")

        td = row.get("tokens_emitted_delta")
        wd = row.get("wall_time_s_delta")
        ed = row.get("avg_entropy_delta")

        lines.append("<table>")
        lines.append("<thead><tr><th>Metric</th><th>Baseline</th><th>SRB</th><th>Δ (SRB - Base)</th><th>% Change</th></tr></thead><tbody>")
        lines.append(f"<tr><td>Tokens emitted</td><td>{f(bt,0)}</td><td>{f(st,0)}</td><td>{f(td,0)}</td><td>{pct(st, bt)}</td></tr>")
        lines.append(f"<tr><td>Wall time (s)</td><td>{f(bw)}</td><td>{f(sw)}</td><td>{f(wd)}</td><td>{pct(sw, bw)}</td></tr>")
        lines.append(f"<tr><td>Avg entropy</td><td>{f(be)}</td><td>{f(se)}</td><td>{f(ed)}</td><td>{pct(se, be)}</td></tr>")
        lines.append(f"<tr><td>Stop reason</td><td>{esc(row.get('base_stop_reason', '-') or '-')}</td><td>{esc(row.get('srb_stop_reason', '-') or '-')}</td><td>—</td><td>—</td></tr>")
        lines.append("</tbody></table>")

        # Responses
        base_txt = row.get("base_response_text", "")
        srb_txt  = row.get("srb_response_text", "")

        base_txt = "" if pd.isna(base_txt) else str(base_txt)
        srb_txt  = "" if pd.isna(srb_txt)  else str(srb_txt)

        if truncate_response and truncate_response > 0:
            base_txt = base_txt[:truncate_response] + ("…" if len(base_txt) > truncate_response else "")
            srb_txt  = srb_txt[:truncate_response]  + ("…" if len(srb_txt)  > truncate_response else "")

        lines.append("<details><summary>Baseline response</summary>")
        lines.append(f"<pre>{esc(base_txt)}</pre></details>")
        lines.append("<details><summary>SRB response</summary>")
        lines.append(f"<pre>{esc(srb_txt)}</pre></details>")
        lines.append("</div>")  # .card

    lines.append("<div class='footer'>Generated by <code class='k'>format_runs_longform.py</code>. Share as-is or export to PDF.</div>")
    lines.append("</body></html>")
    return "\n".join(lines)


def main():
    args = parse_args()
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"ERROR: Input not found: {inp}")

    df = pd.read_csv(inp)
    validate_columns(df)

    # Normalize strategy labels
    df["strategy"] = df["strategy"].str.strip().str.lower()

    # Keep only known strategies
    df = df[df["strategy"].isin(["static", "srb_dynamic"])].copy()

    # Build a key to pair rows
    df["pair_key"] = list(zip(df["bucket"], df["name"], df["prompt_ix"]))

    # Split then merge to create side-by-side baseline vs SRB
    base = df[df["strategy"] == "static"].copy()
    srb  = df[df["strategy"] == "srb_dynamic"].copy()

    base = base.add_prefix("base_")
    srb  = srb.add_prefix("srb_")

    merged = pd.merge(
        base,
        srb,
        left_on="base_pair_key",
        right_on="srb_pair_key",
        how="outer",
        validate="one_to_one"
    )

    # Derive “shared” fields from whichever side is present
    def choose(a, b):
        return a if pd.notna(a) else b

    merged["bucket"]    = merged.apply(lambda r: choose(r.get("base_bucket"), r.get("srb_bucket")), axis=1)
    merged["name"]      = merged.apply(lambda r: choose(r.get("base_name"), r.get("srb_name")), axis=1)
    merged["prompt_ix"] = merged.apply(lambda r: choose(r.get("base_prompt_ix"), r.get("srb_prompt_ix")), axis=1)
    merged["prompt"]    = merged.apply(lambda r: choose(r.get("base_prompt"), r.get("srb_prompt")), axis=1)

    # Compute deltas & percent changes
    merged["tokens_emitted_delta"] = merged["srb_tokens_emitted"] - merged["base_tokens_emitted"]
    merged["wall_time_s_delta"]    = merged["srb_wall_time_s"]    - merged["base_wall_time_s"]
    merged["avg_entropy_delta"]    = merged["srb_avg_entropy"]    - merged["base_avg_entropy"]

    merged["tokens_emitted_pct"] = (merged["tokens_emitted_delta"] / merged["base_tokens_emitted"]) * 100
    merged["wall_time_s_pct"]    = (merged["wall_time_s_delta"]    / merged["base_wall_time_s"])    * 100
    merged["avg_entropy_pct"]    = (merged["avg_entropy_delta"]    / merged["base_avg_entropy"])    * 100

    # Sort for stable report
    merged.sort_values(["bucket", "name", "prompt_ix"], inplace=True, kind="stable")

    # Optional: write paired CSV
    if args.paired_csv:
        out_csv = Path(args.paired_csv)
        keep_cols = [
            "bucket","name","prompt_ix","prompt",
            "base_tokens_emitted","srb_tokens_emitted","tokens_emitted_delta","tokens_emitted_pct",
            "base_wall_time_s","srb_wall_time_s","wall_time_s_delta","wall_time_s_pct",
            "base_avg_entropy","srb_avg_entropy","avg_entropy_delta","avg_entropy_pct",
            "base_stop_reason","srb_stop_reason"
        ]
        available = [c for c in keep_cols if c in merged.columns]
        merged[available].to_csv(out_csv, index=False)

    # Build Markdown
    lines = []
    lines.append("# SRB Longform Report\n")
    lines.append(f"- **Source:** `{inp.name}`")
    lines.append(f"- **Total prompts paired:** {merged.shape[0]}")
    lines.append("")

    # Overview by bucket
    bucket_counts = merged["bucket"].value_counts().to_dict()
    if bucket_counts:
        lines.append("## Overview by bucket")
        for b, n in bucket_counts.items():
            lines.append(f"- **{b}**: {n} prompts")
        lines.append("")

    # Per-prompt sections
    last_bucket = None
    for _, row in merged.iterrows():
        bucket = row.get("bucket", "unknown")
        name   = row.get("name", "unknown")
        pidx   = row.get("prompt_ix", "-")
        prompt = row.get("prompt", "")

        if bucket != last_bucket:
            lines.append(f"\n## {bucket}\n")
            last_bucket = bucket

        # Compact header
        lines.append(f"### `{name}` (idx: {pidx})")
        if isinstance(prompt, str) and prompt.strip():
            wrapped = fill(prompt.strip(), width=args.wrap_width)
            lines.append(f"> {wrapped}")
        lines.append("")

        # Metrics table
        def f(v, d=3): return format_float(v, d)

        table = [
"| Metric | Baseline | SRB | Δ (SRB - Base) | % Change |",
"|---|---:|---:|---:|---:|",
f"| Tokens emitted | {f(row.get('base_tokens_emitted'),0)} | {f(row.get('srb_tokens_emitted'),0)} | {f(row.get('tokens_emitted_delta'),0)} | {pct_change(row.get('srb_tokens_emitted'), row.get('base_tokens_emitted'))} |",
f"| Wall time (s)  | {f(row.get('base_wall_time_s'))} | {f(row.get('srb_wall_time_s'))} | {f(row.get('wall_time_s_delta'))} | {pct_change(row.get('srb_wall_time_s'), row.get('base_wall_time_s'))} |",
f"| Avg entropy    | {f(row.get('base_avg_entropy'))} | {f(row.get('srb_avg_entropy'))} | {f(row.get('avg_entropy_delta'))} | {pct_change(row.get('srb_avg_entropy'), row.get('base_avg_entropy'))} |",
f"| Stop reason    | {row.get('base_stop_reason', '-') or '-'} | {row.get('srb_stop_reason', '-') or '-'} | — | — |",
        ]
        lines.extend(table)
        lines.append("")

        # Full responses in collapsible sections
        base_txt = row.get("base_response_text", "")
        srb_txt  = row.get("srb_response_text", "")

        # If the CSV columns were prefixed by us, extract them from the merged frame
        if pd.isna(base_txt) and "base_response_text" not in merged.columns and "response_text" in df.columns:
            # Fallback: try to map from original df (should not be needed)
            pass

        base_txt = "" if pd.isna(base_txt) else str(base_txt)
        srb_txt  = "" if pd.isna(srb_txt)  else str(srb_txt)

        base_txt = safe_truncate(base_txt, args.truncate_response)
        srb_txt  = safe_truncate(srb_txt,  args.truncate_response)

        # Use fenced code blocks for readability inside details
        lines.append("<details>")
        lines.append("<summary><strong>Baseline response</strong></summary>\n")
        lines.append("```text")
        lines.append(base_txt)
        lines.append("```\n</details>\n")

        lines.append("<details>")
        lines.append("<summary><strong>SRB response</strong></summary>\n")
        lines.append("```text")
        lines.append(srb_txt)
        lines.append("```\n</details>\n")

        lines.append("---\n")

    out_md = Path(args.output)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote Markdown report: {out_md}")

    # Optionally write HTML
    if args.html_output:
        html_text = build_html_report(
            merged=merged,
            source_name=inp.name,
            wrap_width=args.wrap_width,
            truncate_response=args.truncate_response,
        )
        out_html = Path(args.html_output)
        out_html.write_text(html_text, encoding="utf-8")
        print(f"Wrote HTML report: {out_html}")


if __name__ == "__main__":
    # Ensure pandas is present; graceful message if not
    try:
        import pandas as _check  # noqa
    except Exception:
        sys.stderr.write("This script requires pandas. Try: pip install -r requirements.txt\n")
        sys.exit(1)
    main()