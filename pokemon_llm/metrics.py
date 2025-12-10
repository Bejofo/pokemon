import os
import glob
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

VALID_TYPES = [
    "Normal","Fire","Water","Electric","Grass","Ice","Fighting","Poison",
    "Ground","Flying","Psychic","Bug","Rock","Ghost","Dragon","Dark",
    "Steel","Fairy"
]

def compute_type_metrics(df):
    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        true_set = set(
            t for t in [row["type1"], row["type2"]]
            if isinstance(t, str) and t.strip()
        )
        pred_set = set(
            t for t in [row["predicted_type1"], row["predicted_type2"]]
            if isinstance(t, str) and t.strip()
        )

        true_vec = [1 if t in true_set else 0 for t in VALID_TYPES]
        pred_vec = [1 if t in pred_set else 0 for t in VALID_TYPES]

        y_true.append(true_vec)
        y_pred.append(pred_vec)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    

    return round(precision, 3), round(recall, 3), round(f1, 3)


if __name__ == "__main__":
    output_dir = "pokemon_llm/outputs"
    csv_files = glob.glob(os.path.join(output_dir, "results_*.csv"))

    all_metrics = []

    for path in csv_files:
        df = pd.read_csv(path)

        binary_accuracy = round(df["binary_match_score"].mean(), 3)
        partial_accuracy = round(df["partial_match_score"].mean(), 3)
        precision, recall, f1 = compute_type_metrics(df)

        model_name = os.path.basename(path)
        model_name = model_name.replace("results_", "").replace(".csv", "")

        all_metrics.append({
            "model": model_name,
            "binary_accuracy": binary_accuracy,
            "partial_accuracy": partial_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    global_df = pd.DataFrame(all_metrics)
    global_df.to_csv("pokemon_llm/results/metrics.csv", index=False)
