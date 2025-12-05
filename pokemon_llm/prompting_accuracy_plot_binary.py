import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUTPUT_DIR = "pokemon_llm/outputs"
SAVE_PATH = "pokemon_llm/results/prompting_accuracy_plot_binary.png"
PROMPTS = ["zero-shot", "cot", "few-shot"]

def extract_raw_model_from_filename(path: str) -> str:
    name = os.path.basename(path)
    if name.startswith("results_"):
        name = name[len("results_"):]
    if name.endswith(".csv"):
        name = name[:-4]
    return name

def extract_prompt(rawname: str) -> str:
    rn = rawname.lower()
    if "_few_shot" in rn:
        return "few-shot"
    if "_zero_shot_cot" in rn:
        return "cot"
    if "_zero_shot" in rn:
        return "zero-shot"
    return "unknown"

def canonical_model(rawname: str) -> str:
    return re.sub(r'_(zero_shot|zero_shot_cot|few_shot)$',
                  '',
                  rawname,
                  flags=re.IGNORECASE)

rows = []
csv_files = glob.glob(os.path.join(OUTPUT_DIR, "results_*.csv"))
if not csv_files:
    raise SystemExit(f"No results_*.csv files found in {OUTPUT_DIR}")

for path in csv_files:
    rawname = extract_raw_model_from_filename(path)
    prompt = extract_prompt(rawname)
    model = canonical_model(rawname)

    df = pd.read_csv(path)
    # ensure match_score column exists
    if "binary_match_score" not in df.columns:
        raise ValueError(f"File {path} missing 'binary_match_score' column")
    match = float(df["binary_match_score"].mean())

    rows.append({"canonical": model, "prompt": prompt, "match": match, "raw": rawname})

metrics_df = pd.DataFrame(rows)

# Use stable ordering: by canonical name sorted alphabetically.
models = sorted(metrics_df["canonical"].unique())

# If a model is missing for a prompt, use np.nan (will plot as 0 height).
scores_matrix = np.full((len(PROMPTS), len(models)), np.nan, dtype=float)

for i, p in enumerate(PROMPTS):
    for j, m in enumerate(models):
        sel = metrics_df[(metrics_df["prompt"] == p) & (metrics_df["canonical"] == m)]
        if not sel.empty:
            scores_matrix[i, j] = sel["match"].mean()

# Replace NaN with 0.0 so bars draw (absent bars appear at 0)
plot_matrix = np.nan_to_num(scores_matrix, nan=0.0)

cmap = plt.get_cmap("tab20")
colors = [cmap(i % 20) for i in range(len(models))]

fig, ax = plt.subplots(figsize=(10, 5))
n_prompts = len(PROMPTS)
n_models = len(models)

total_width = 0.75
bar_width = total_width / n_models
x_centers = np.arange(n_prompts)

for j, model_name in enumerate(models):
    offsets = x_centers - total_width/2 + j * bar_width + bar_width/2
    values = plot_matrix[:, j]
    ax.bar(offsets, values, width=bar_width * 0.95, color=colors[j], 
    label=model_name, edgecolor="black", linewidth=0.3)
    for x, v in zip(offsets, values):
        if v > 0:
            ax.text(
                x,
                v - 0.03,       # slightly below the top of the bar
                f"{v:.2f}",
                ha="center",
                va="top",
                fontsize=8
            )

for i, p in enumerate(PROMPTS):
    # average across models for this prompt
    avg_val = np.mean(plot_matrix[i, :])
    
    # horizontal span of this bin: center ± total_width/2
    x_left = x_centers[i] - total_width/2
    x_right = x_centers[i] + total_width/2
    
    # draw a darker line (adjust color/alpha as you like)
    ax.hlines(
        y=avg_val,
        xmin=x_left,
        xmax=x_right,
        colors="black",
        linestyles="dashed",
        linewidth=1.5,
        alpha=0.8
    )
    
    ax.text(
        x_centers[i] - 0.2,   # shift label left
        avg_val + 0.015,      # vertical position
        f"{avg_val:.2f}",
        ha="right",           # anchor text so it sits neatly to the left
        va="bottom",
        fontsize=9
    )

ax.set_xticks(x_centers)
ax.set_xticklabels(PROMPTS, fontsize=12)
ax.set_ylabel("Match Accuracy", fontsize=12)
ax.set_ylim(0, 1.0)
ax.set_title("Match Accuracy by Prompting Method and Model", fontsize=14)
ax.grid(axis="y", linestyle="--", alpha=0.25)

legend_handles = [Patch(facecolor=colors[i], edgecolor="black", label=models[i]) for i in range(len(models))]
ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1.0), loc="upper left", title="Model", frameon=False)

plt.tight_layout()
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=600)