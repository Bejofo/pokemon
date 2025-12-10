import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
root = "pokemon_llm/results"
csv_path = os.path.join(root, "metrics.csv")
df = pd.read_csv(csv_path)
df = df.dropna(subset=["model"])

# -------------------------------------------------------------------
# Parse model names
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
fill_color_map = {m: cmap(i) for i, m in enumerate(base_models)}

x_positions = {pt: i for i, pt in enumerate(prompt_types)}
bar_width = 0.07
model_spacing = 0.04

for pt in prompt_types:
    index = 0
    for base_model in base_models:
        row = df[(df["base_model"] == base_model) & (df["prompting_method"] == pt)]
        if row.empty:
            continue

        f1 = row["f1"].values[0]

        bx = x_positions[pt] + index * (bar_width + model_spacing)
        fill = fill_color_map[base_model]

        # F1 bar
        plt.bar(
            bx,
            f1,
            width=bar_width,
            color=fill
        )

        # Text label above the bar
        plt.text(
            bx,
            f1 + 0.01,
            f"{f1:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

        index += 1


# -------------------------------------------------------------------
# Center x-ticks per prompting method
# -------------------------------------------------------------------
xtick_positions = []
for pt in prompt_types:
    models_in_bin = df[df["prompting_method"] == pt]["base_model"].unique()
    n_models = len(models_in_bin)
    total_width = n_models * bar_width + (n_models - 1) * model_spacing
    xtick_positions.append(x_positions[pt] + total_width/2)

plt.xticks(xtick_positions, prompt_types)
plt.ylabel("F1 Score")
plt.title("F1 Score by Model and Prompting Method")
plt.ylim(0, 1)

# Legend: model fill colors
handles_model = [
    plt.Rectangle((0,0),1,1, facecolor=fill_color_map[m])
    for m in base_models
]
labels_model = list(base_models)
plt.legend(handles_model, labels_model, title="Model", loc="upper left", bbox_to_anchor=(0.15, 1.0))

plt.tight_layout()

# Save figure
output_path = os.path.join(root, "f1_plot.png")
plt.savefig(output_path, dpi=600)
