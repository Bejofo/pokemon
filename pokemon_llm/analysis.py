import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "pokemon_llm/outputs"

dfs = []
for file in os.listdir(OUTPUT_DIR):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(OUTPUT_DIR, file)
    name = os.path.splitext(file)[0]

    # Infer prompt type and model name from filename
    prompt_type = next(
        (p for p in ["zero_shot_cot", "zero_shot", "few_shot", "cot", "zero", "few"] if p in name),
        "unknown",
    )
    model_name = name.replace(prompt_type, "").replace("__", "_").strip("_")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]  # normalize column names

    # Identify the match score column dynamically
    score_col = next((c for c in df.columns if "match" in c and "score" in c), None)
    if score_col is None:
        print(f"⚠️ Skipping {file}: no match_scores column found.")
        continue

    df["model"] = model_name
    df["prompt_type"] = prompt_type
    df["accuracy"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)

    dfs.append(df)

if not dfs:
    raise RuntimeError("No valid CSVs with match_scores column found in outputs folder.")

data = pd.concat(dfs, ignore_index=True)

# --- BASIC SUMMARY ---
print("=== Summary by Model and Prompt ===")
print(data.groupby(["model", "prompt_type"])["accuracy"].describe(), "\n")

# --- CONVERT MATCH SCORES TO CATEGORIES ---
score_map = { 0.0: "completely_wrong (1 or 2 types)", 
             0.25: "one_correct_wrong_order (2 types)", 
             0.5: "one_correct_right_order (2 types)", 
             0.75: "both_correct_wrong_order (2 types)", 
             1.0: "completely_correct (1 or 2 types)", }
data["accuracy_label"] = data["accuracy"].map(score_map).fillna("unknown")

# --- PLOTS ---

# 1. Which prompting method performed best across models
plt.figure(figsize=(10, 5))
pivot_prompt = data.groupby(["prompt_type", "model"])["accuracy"].mean().unstack()
pivot_prompt.plot(kind="bar", figsize=(10, 5))
plt.title("Mean Accuracy by Prompt Type Across Models")
plt.ylabel("Mean Accuracy")
plt.xlabel("Prompt Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_by_prompt_model.png", dpi=200)
plt.close()

# 2. Which models performed best overall
plt.figure(figsize=(8, 4))
data.groupby("model")["accuracy"].mean().sort_values(ascending=False).plot(kind="bar")
plt.title("Average Accuracy by Model")
plt.ylabel("Mean Accuracy")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_by_model.png", dpi=200)
plt.close()

# 3. Accuracy distribution by model and prompt type
plt.figure(figsize=(12, 6))
data.boxplot(column="accuracy", by=["model", "prompt_type"], grid=False, rot=45)
plt.title("Accuracy Distribution by Model and Prompt Type")
plt.suptitle("")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_boxplot.png", dpi=200)
plt.close()

# 4. Mean accuracy per primary Pokémon type (if available)
if "type1" in data.columns:
    type_acc = (
        data.groupby("type1")["accuracy"]
        .mean()
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(10, 4))
    type_acc.plot(kind="bar")
    plt.title("Mean Accuracy by Primary Pokémon Type (type1)")
    plt.ylabel("Mean Accuracy")
    plt.xlabel("Type")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/accuracy_by_type1.png", dpi=200)
    plt.close()

# 5. Misclassification correlation (if available)
if {"type1", "predicted_type1"} <= set(data.columns):
    wrong = data[data["accuracy"] < 1]
    conf_matrix = pd.crosstab(
        wrong["type1"], wrong["predicted_type1"], normalize="index"
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap="Reds", aspect="auto")
    plt.colorbar(label="Proportion of Misclassifications")
    plt.title("Misclassification Correlation: True vs Predicted (Type1)")
    plt.xticks(range(len(conf_matrix.columns)), conf_matrix.columns, rotation=90)
    plt.yticks(range(len(conf_matrix.index)), conf_matrix.index)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/misclassification_correlation.png", dpi=200)
    plt.close()

# 6. Frequency of each categorical match score
plt.figure(figsize=(7, 4))
data["accuracy_label"].value_counts(normalize=True).sort_index().plot(kind="bar")
plt.title("Distribution of Match Score Categories")
plt.ylabel("Proportion")
plt.xlabel("Match Score Category")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/match_score_distribution.png", dpi=200)
plt.close()

print("✅ Analysis complete. Plots saved in 'pokemon_llm/outputs/' folder.")
