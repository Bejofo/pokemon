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

# Unique models
models = df["model"].unique()

plt.figure(figsize=(12, 6))

# Color map for models
cmap = plt.get_cmap("tab20")
color_map = {m: cmap(i) for i, m in enumerate(models)}

# Layout parameters
bar_width = 0.3
x = range(len(models))

# -------------------------------------------------------------------
# Draw bars
# -------------------------------------------------------------------
for i, model in enumerate(models):
    row = df[df["model"] == model]
    if row.empty:
        continue

    binary = row["binary_accuracy"].iloc[0]
    partial = row["partial_accuracy"].iloc[0]

    # Binary bar position
    x_bin = i - bar_width/2

    # Partial bar position
    x_par = i + bar_width/2

    # Binary accuracy bar
    plt.bar(
        x_bin,
        binary,
        width=bar_width,
        color=color_map[model]
    )

    # Text label for binary
    plt.text(
        x_bin,
        binary + 0.01,
        f"{binary:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

    # Partial accuracy bar (hatched)
    plt.bar(
        x_par,
        partial,
        width=bar_width,
        color=color_map[model],
        hatch='//'
    )

    # Text label for partial
    plt.text(
        x_par,
        partial + 0.01,
        f"{partial:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )


# -------------------------------------------------------------------
# Labels, ticks, legends
# -------------------------------------------------------------------
plt.xticks([])         
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Binary and Partial Accuracy per Model")
plt.ylim(0, 1)

# Model legend
import matplotlib.patches as mpatches
model_handles = [
    mpatches.Patch(facecolor=color_map[m], label=m)
    for m in models
]

metric_binary = mpatches.Patch(facecolor='gray', label="Binary Accuracy")
metric_partial = mpatches.Patch(facecolor='gray', hatch='//', label="Partial Accuracy")

legend1 = plt.legend(handles=model_handles, title="Model", loc="upper left")
plt.legend(handles=[metric_binary, metric_partial], title="Metric", loc="upper right")
plt.gca().add_artist(legend1)

plt.tight_layout()

# Save figure
output_path = os.path.join(root, "accuracy_plot.png")
plt.savefig(output_path, dpi=600)
