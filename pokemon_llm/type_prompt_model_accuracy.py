import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

OUTPUT_DIR = "pokemon_llm/outputs"
SAVE_PATH = "pokemon_llm/results/per_type_prompt_accuracy.png"

VALID_TYPES = [
    "Normal","Fire","Water","Electric","Grass","Ice","Fighting","Poison",
    "Ground","Flying","Psychic","Bug","Rock","Ghost","Dragon","Dark",
    "Steel","Fairy"
]

PROMPTS = ["zero_shot", "cot", "few_shot"]

def extract_prompt(model_name):
    name = model_name.lower()
    if "few_shot" in name:
        return "few_shot"
    if "zero_shot_cot" in name:
        return "cot"
    if "zero_shot" in name:
        return "zero_shot"
    return "unknown"


# per-type accuracy: { prompt: { type: [correct_flags] } }
stats = {p: {t: [] for t in VALID_TYPES} for p in PROMPTS}

csv_files = glob.glob(os.path.join(OUTPUT_DIR, "results_*.csv"))
if not csv_files:
    raise SystemExit("No result files found.")

for path in csv_files:
    df = pd.read_csv(path)
    rawname = os.path.basename(path)
    prompt = extract_prompt(rawname)

    if prompt not in PROMPTS:
        continue

    for _, row in df.iterrows():
        true_types = set(t for t in [row["type1"], row["type2"]] if isinstance(t, str) and t.strip())
        pred_types = set(t for t in [row["predicted_type1"], row["predicted_type2"]] if isinstance(t, str) and t.strip())

        for t in VALID_TYPES:
            # 1 if correctly predicted presence/absence of this type
            correct = int((t in true_types) == (t in pred_types))
            stats[prompt][t].append(correct)


# compute average accuracy per type per prompt
accuracy = {p: {t: (sum(vals)/len(vals) if vals else 0.0)
                for t, vals in stats[p].items()}
            for p in PROMPTS}

# plot
fig, ax = plt.subplots(figsize=(14,6))

x = range(len(VALID_TYPES))
for p, marker in zip(PROMPTS, ["o", "s", "^"]):
    y = [accuracy[p][t] for t in VALID_TYPES]
    ax.plot(x, y, marker=marker, label=p.replace("_", "-"), linewidth=2)

ax.set_xticks(x)
ax.set_xticklabels(VALID_TYPES, rotation=45, ha="right")
ax.set_ylabel("Per-Type Accuracy")
ax.set_title("Per-Type Accuracy by Prompting Method (Aggregated Across All Models)")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.3)

ax.legend(title="Prompting Method")

plt.tight_layout()
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300)
