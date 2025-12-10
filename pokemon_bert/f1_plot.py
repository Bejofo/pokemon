import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
root = "pokemon_bert/results"
csv_path = os.path.join(root, "metrics.csv")
df = pd.read_csv(csv_path)
df = df.dropna(subset=["model"])

# -------------------------------------------------------------------
# Unique models
# -------------------------------------------------------------------
models = df["model"].unique()

plt.figure(figsize=(12, 6))

# Color map
cmap = plt.get_cmap("tab20")
color_map = {m: cmap(i) for i, m in enumerate(models)}

# X positions
x = range(len(models))
bar_width = 0.5

# -------------------------------------------------------------------
# Draw F1 score bars
# -------------------------------------------------------------------
for i, model in enumerate(models):
    row = df[df["model"] == model]
    if row.empty:
        continue

    f1 = row["f1"].iloc[0]

    plt.bar(i, f1, width=bar_width, color=color_map[model])

    # Add text label above the bar
    plt.text(
        i,                   # x-position (same as bar center)
        f1 + 0.01,           # y-position slightly above bar
        f"{f1:.3f}",         # formatted score
        ha="center", 
        va="bottom",
        fontsize=9
    )

# -------------------------------------------------------------------
# Labels and ticks
# -------------------------------------------------------------------
plt.xticks([])         
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.title("F1 Score per Model")
plt.ylim(0, 1)

# Legend
import matplotlib.patches as mpatches
handles = [mpatches.Patch(facecolor=color_map[m], label=m) for m in models]
plt.legend(handles, models, title="Model", loc="upper left")

plt.tight_layout()

# Save
output_path = os.path.join(root, "f1_plot.png")
plt.savefig(output_path, dpi=600)
