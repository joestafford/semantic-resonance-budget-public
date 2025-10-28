# semantic-resonance-budget-public
Public sharing repository for Semantic Resonance Budget findings

## Overview
The Semantic Resonance Budget (SRB) is a novel inference-time method designed to dynamically adjust the generation process based on semantic entropy. By modulating the generation length and content adaptively, SRB aims to improve response coherence and reduce wall-clock time without sacrificing, and often enhancing, the overall quality of generated outputs. This repository provides data and results from experiments evaluating the effectiveness of SRB in longform benchmark tasks.

## Conceptual Summary

The **Semantic Resonance Budget (SRB)** is an inference-time optimization method rooted in the study of *semantic thermodynamics* — the relationship between meaning, uncertainty, and energy expenditure in language generation. Rather than fixing a static number of tokens or relying solely on probability thresholds, SRB continuously monitors **semantic entropy** across each generation step to determine when a response has reached its point of maximal coherence.

In practice, SRB measures how rapidly the model’s semantic uncertainty changes and dynamically adjusts the continuation process. When entropy stabilizes (indicating that additional tokens are adding little new meaning), SRB signals an optimal stopping point. Conversely, when entropy oscillates or rises meaningfully, SRB allows continuation to capture unfolding ideas or narrative expansions.

This behavior can yield multiple benefits:
- **Reduced wall-clock time** in responses, since the model stops early when semantic completion is reached.
- **Maintained or improved response quality**, as SRB prevents both premature truncation and excessive rambling.
- **Dynamic adaptation** to different prompt types — concise factual answers stabilize early, while complex reasoning or storytelling prompts extend naturally.

At a theoretical level, SRB demonstrates that semantic information behaves analogously to thermodynamic energy: it flows, stabilizes, and dissipates as meaning is expressed. This connection grounds SRB within broader research on **information physics** and **entropy-aware computation**, offering a bridge between linguistic coherence and measurable physical analogs.

## Included Files
- `summary_longform.csv`: This file contains a summary of the key results from the longform benchmark experiment, including metrics related to response quality, coherence, and computational efficiency.
- `runs_longform.txt`: This file logs detailed run information and parameters used during the longform benchmark experiments, facilitating reproducibility and deeper analysis.

## Usage
Users interested in exploring the results can open the CSV files in spreadsheet software or visualization tools to analyze the data. If you utilize this repository or reference its findings in your research or discussions, please cite it accordingly to acknowledge the work behind the Semantic Resonance Budget methodology.

## Setup Instructions

To reproduce the analyses or generate visualizations, you can set up a local Python environment as follows:

### 1. Create a Python virtual environment
```bash
python3 -m venv .venv
```

### 2. Activate the virtual environment
- **macOS / Linux**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell)**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

### 3. Install dependencies
Once the virtual environment is active, install all required packages using:
```bash
pip install -r requirements.txt
```

### 4. Run the visualization scripts
Each visualization script can be run directly, for example:
```bash
python plot_entropy_overlay.py
```

This will generate the corresponding visual artifact in your working directory.
