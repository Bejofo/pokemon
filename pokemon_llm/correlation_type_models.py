import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "pokemon_llm/outputs"
SAVE_PATH = "pokemon_llm/results/type_correlation_matrix.png"

VALID_TYPES = [
    "Normal","Fire","Water","Electric","Grass","Ice","Fighting","Poison",
    "Ground","Flying","Psychic","Bug","Rock","Ghost","Dragon","Dark",
    "Steel","Fairy"
]

def extract_types(row):
    true_set = set(t for t in [row["type1"], row["type2"]] if isinstance(t, str) and t.strip())
    pred_set = set(t for t in [row["predicted_type1"], row["predicted_type2"]] if isinstance(t, str) and t.strip())
    return true_set, pred_set


# Collect correctness per type per sample across all models/prompts
records = []

csv_files = glob.glob(os.path.join(OUTPUT_DIR, "results_*.csv"))
if not csv_files:
    raise SystemExit("No result files found.")

for path in csv_files:
    df = pd.read_csv(path)

    for _, row in df.iterrows():
        true_set, pred_set = extract_types(row)

        correctness = {}
        for t in VALID_TYPES:
            correctness[t] = int((t in true_set) == (t in pred_set))

        records.append(correctness)

# Build DataFrame: rows = samples (× all models), cols = types
corr_df = pd.DataFrame(records)

# Compute correlation matrix
corr_matrix = corr_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    vmin=-1, vmax=1,
    annot=False,
    xticklabels=True,
    yticklabels=True
)

plt.title("Correlation of Type-Level Correctness Across All Models and Prompts")
plt.tight_layout()

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300)
