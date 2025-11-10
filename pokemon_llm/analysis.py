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

# --- PLOTS ---

# 1. Accuracy distribution by model and prompt type
plt.figure(figsize=(10, 5))
data.boxplot(column="accuracy", by=["model", "prompt_type"], grid=False, rot=45)
plt.title("Accuracy Distribution by Model and Prompt Type")
plt.suptitle("")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_by_model_prompt.png", dpi=200)
plt.close()

# 2. Mean accuracy per true type (if available)
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

# 3. Most common incorrectly predicted types (if available)
if {"predicted_type1", "predicted_type2"} <= set(data.columns):
    wrong = data[data["accuracy"] < 1]
    pred_types = pd.concat([wrong["predicted_type1"], wrong["predicted_type2"]]).dropna()
    if not pred_types.empty:
        plt.figure(figsize=(10, 4))
        pred_types.value_counts().head(15).plot(kind="bar")
        plt.title("Most Frequent Wrongly Predicted Types")
        plt.ylabel("Count")
        plt.xlabel("Predicted Type")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/wrong_predictions.png", dpi=200)
        plt.close()

# 4. Average accuracy by prompt type
plt.figure(figsize=(6, 4))
data.groupby("prompt_type")["accuracy"].mean().plot(kind="bar")
plt.title("Average Accuracy by Prompt Type")
plt.ylabel("Mean Accuracy")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_by_prompt_type.png", dpi=200)
plt.close()

print("✅ Analysis complete. Plots saved in 'pokemon_llm/outputs/' folder.")
