import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
root = "pokemon_llm/results"
csv_path = os.path.join(root, "metrics.csv")
df = pd.read_csv(csv_path)

df = df.dropna(subset=["model"])

# -------------------------------------------------------------------
# Correct parsing of model names
# -------------------------------------------------------------------
def split_model(m):
    if m.endswith("_zero_shot_cot"):
        base = m[:-len("_zero_shot_cot")]
        return base, "zero_shot_cot"
    elif m.endswith("_zero_shot"):
        base = m[:-len("_zero_shot")]
        return base, "zero_shot"
    elif m.endswith("_few_shot"):
        base = m[:-len("_few_shot")]
        return base, "few_shot"
    else:
        parts = m.split("_")
        return "_".join(parts[:-1]), parts[-1]

df[["base_model", "prompting_method"]] = df["model"].apply(
    lambda m: pd.Series(split_model(m))
)

prompt_types = ["zero_shot", "zero_shot_cot", "few_shot"]
df = df[df["prompting_method"].isin(prompt_types)]

# -------------------------------------------------------------------
# Plot setup
# -------------------------------------------------------------------
plt.figure(figsize=(12, 6))

base_models = df["base_model"].unique()
cmap = plt.get_cmap("tab20")

# Fill = model identity
fill_color_map = {m: cmap(i) for i, m in enumerate(base_models)}

x_positions = {
    "zero_shot": 0,
    "zero_shot_cot": 1.2,   # extra space
    "few_shot": 2.4          # extra space
}

# Widths and spacing
bar_width = 0.05           # narrower bars
model_spacing = 0.04       # gap between models
metric_spacing = 0.015     # small gap between binary + partial bars for same model

for pt in prompt_types:
    index = 0
    for base_model in base_models:

        row = df[(df["base_model"] == base_model) & (df["prompting_method"] == pt)]
        if row.empty:
            continue

        # Base x for the model within its prompting-method group
        base_x = x_positions[pt] + index * (2*bar_width + metric_spacing + model_spacing)
        fill = fill_color_map[base_model]

        # Binary bar — solid fill, no outline
        plt.bar(
            base_x,
            row["binary_accuracy"].values[0],
            width=bar_width,
            color=fill
        )

        # Partial bar — diagonally striped fill, no outline
        plt.bar(
            base_x + bar_width + metric_spacing,
            row["partial_accuracy"].values[0],
            width=bar_width,
            color=fill,
            hatch='//'
        )

        index += 1

# -------------------------------------------------------------------
# Labels, ticks, legends
# -------------------------------------------------------------------
xtick_positions = []
for pt in prompt_types:
    models_in_bin = df[df["prompting_method"] == pt]["base_model"].unique()
    n_models = len(models_in_bin)
    total_width = n_models * (2*bar_width + metric_spacing + model_spacing) - model_spacing
    xtick_positions.append(x_positions[pt] + total_width/2)

plt.xticks(xtick_positions, prompt_types)
plt.ylabel("Accuracy")
plt.title("Binary and Partial Accuracy by Model and Prompting Method")
plt.ylim(0, 1)

# Model legend (fill colors)
handles_model = [
    plt.Rectangle((0, 0), 1, 1, facecolor=fill_color_map[m])
    for m in base_models
]
labels_model = list(base_models)
legend1 = plt.legend(handles_model, labels_model, title="Model", loc="upper left")

# Metric legend (solid vs hatched)
import matplotlib.patches as mpatches
binary_patch = mpatches.Patch(facecolor='gray', label='Binary Accuracy')
partial_patch = mpatches.Patch(facecolor='gray', hatch='//', label='Partial Accuracy')

plt.legend(handles=[binary_patch, partial_patch], title="Metric", loc="upper right")
plt.gca().add_artist(legend1)

plt.tight_layout()

# Save figure
output_path = os.path.join(root, "accuracy_plot.png")
plt.savefig(output_path, dpi=300)
