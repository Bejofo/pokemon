import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
root = "pokemon_ngram/results"
csv_path = os.path.join(root, "metrics.csv")
df = pd.read_csv(csv_path)
df = df.dropna(subset=["model"])

# -------------------------------------------------------------------
# Parse model into vectorizer + classifier
# -------------------------------------------------------------------
def split_model_to_vec_clf(m):
    parts = str(m).split("_")
    if len(parts) < 3:
        return m, m
    vectorizer = "_".join(parts[:2])
    classifier = "_".join(parts[2:])
    return vectorizer, classifier

df[["vectorizer", "classifier"]] = df["model"].apply(
    lambda m: pd.Series(split_model_to_vec_clf(m))
)

vectorizer_bins = ["CountVectorizer_1gram", "TfidfVectorizer_1gram"]
df = df[df["vectorizer"].isin(vectorizer_bins)]

# -------------------------------------------------------------------
# Plot setup
# -------------------------------------------------------------------
plt.figure(figsize=(12, 6))

classifiers = df["classifier"].unique()
cmap = plt.get_cmap("tab20")

# one fill color per classifier
fill_color_map = {clf: cmap(i) for i, clf in enumerate(classifiers)}

# x positions for the two bins
x_positions = {
    "CountVectorizer_1gram": 0,
    "TfidfVectorizer_1gram": 0.5
}

bar_width = 0.07
model_spacing = 0.04

for vec in vectorizer_bins:
    index = 0
    for clf in classifiers:
        row = df[(df["vectorizer"] == vec) & (df["classifier"] == clf)]
        if row.empty:
            continue

        f1 = float(row["f1"].values[0])

        bx = x_positions[vec] + index * (bar_width + model_spacing)
        fill = fill_color_map[clf]

        # F1 bar
        plt.bar(
            bx,
            f1,
            width=bar_width,
            color=fill
        )

        # text label
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
# Center x-ticks under each bin
# -------------------------------------------------------------------
xtick_positions = []
for vec in vectorizer_bins:
    clfs_in_bin = df[df["vectorizer"] == vec]["classifier"].unique()
    n_models = len(clfs_in_bin)
    total_width = n_models * bar_width + (n_models - 1) * model_spacing
    xtick_positions.append(x_positions[vec] + total_width/2)

plt.xticks(xtick_positions, ["CountVectorizer", "TfidfVectorizer"])
plt.ylabel("F1 Score")
plt.title("F1 Score by Vectorizer and Classifier")
plt.ylim(0, 1)

# -------------------------------------------------------------------
# Legend
# -------------------------------------------------------------------
handles_model = [
    plt.Rectangle((0, 0), 1, 1, facecolor=fill_color_map[c])
    for c in classifiers
]
plt.legend(handles_model, list(classifiers), title="Classifier",
           loc="upper left")

plt.tight_layout()

# Save
output_path = os.path.join(root, "f1_plot.png")
plt.savefig(output_path, dpi=600)
