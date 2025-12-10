import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
root = "pokemon_ngram/results"
csv_path = os.path.join(root, "metrics.csv")
df = pd.read_csv(csv_path)

df = df.dropna(subset=["model"])

# -------------------------------------------------------------------
# Parse model into vectorizer + classifier
# Example: "CountVectorizer_1gram_KNeighborsClassifier"
# -> vectorizer = "CountVectorizer_1gram"
# -> classifier = "KNeighborsClassifier"
# -------------------------------------------------------------------
def split_model_to_vec_clf(m):
    parts = str(m).split("_")
    if len(parts) < 3:
        # fallback: treat whole as classifier, empty vectorizer
        return m, m
    vectorizer = "_".join(parts[:2])
    classifier = "_".join(parts[2:])
    return vectorizer, classifier

df[["vectorizer", "classifier"]] = df["model"].apply(
    lambda m: pd.Series(split_model_to_vec_clf(m))
)

# Keep only the two vectorizer bins we care about (and preserve order)
vectorizer_bins = ["CountVectorizer_1gram", "TfidfVectorizer_1gram"]
df = df[df["vectorizer"].isin(vectorizer_bins)]

# -------------------------------------------------------------------
# Plot setup
# -------------------------------------------------------------------
plt.figure(figsize=(12, 6))

classifiers = df["classifier"].unique()
cmap = plt.get_cmap("tab20")

# Fill = classifier identity (consistent across bins)
fill_color_map = {m: cmap(i) for i, m in enumerate(classifiers)}

# X positions for vectorizer bins
x_positions = {
    "CountVectorizer_1gram": 0,
    "TfidfVectorizer_1gram": 0.6  # extra space between bins
}

# Widths and spacing
bar_width = 0.05           # narrower bars
model_spacing = 0.04       # gap between classifiers
metric_spacing = 0.015     # small gap between binary + partial bars for same classifier

for vec in vectorizer_bins:
    index = 0
    for clf in classifiers:

        row = df[(df["vectorizer"] == vec) & (df["classifier"] == clf)]
        if row.empty:
            continue

        binary = float(row["binary_accuracy"].values[0])
        partial = float(row["partial_accuracy"].values[0])

        base_x = x_positions[vec] + index * (2*bar_width + metric_spacing + model_spacing)
        fill = fill_color_map[clf]

        # Binary bar
        plt.bar(
            base_x,
            binary,
            width=bar_width,
            color=fill
        )

        plt.text(
            base_x,
            binary + 0.01,
            f"{binary:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

        # Partial bar
        px = base_x + bar_width + metric_spacing
        plt.bar(
            px,
            partial,
            width=bar_width,
            color=fill,
            hatch='//'
        )

        plt.text(
            px,
            partial + 0.01,
            f"{partial:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

        index += 1

# -------------------------------------------------------------------
# Labels, ticks, legends
# -------------------------------------------------------------------
xtick_positions = []
for vec in vectorizer_bins:
    classifiers_in_bin = df[df["vectorizer"] == vec]["classifier"].unique()
    n_models = len(classifiers_in_bin)
    total_width = n_models * (2*bar_width + metric_spacing + model_spacing) - model_spacing
    xtick_positions.append(x_positions[vec] + total_width/2)

plt.xticks(xtick_positions, ["CountVectorizer", "TfidfVectorizer"])
plt.ylabel("Accuracy")
plt.title("Binary and Partial Accuracy by Vectorizer and Classifier")
plt.ylim(0, 1)

# Classifier legend (fill colors)
import matplotlib.patches as mpatches
handles_model = [
    plt.Rectangle((0, 0), 1, 1, facecolor=fill_color_map[m])
    for m in classifiers
]
labels_model = list(classifiers)
# keep legend inside but nudged right slightly
legend1 = plt.legend(handles_model, labels_model, title="Classifier",
                     loc="upper left")

# Metric legend (solid vs hatched)
binary_patch = mpatches.Patch(facecolor='gray', label='Binary Accuracy')
partial_patch = mpatches.Patch(facecolor='gray', hatch='//', label='Partial Accuracy')

plt.legend(handles=[binary_patch, partial_patch], title="Metric", loc="upper right")
plt.gca().add_artist(legend1)

plt.tight_layout()

# Save figure
output_path = os.path.join(root, "accuracy_plot.png")
plt.savefig(output_path, dpi=600)
