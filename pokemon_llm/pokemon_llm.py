import pandas as pd
import json
import re
import numpy as np
from tqdm import tqdm
import os
import itertools
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class POKEMON_LLM:
    def __init__(self):
        self.data = None
        self.stats = {}
        self.model = None
        self.valid_types = ["Grass", "Water", "Fire", "Electric", "Ice", "Fighting", "Poison", "Dark", "Fairy", "Steel", "Flying", "Normal", "Psychic", "Ghost", "Ground", "Rock", "Dragon", "Bug"]

    def load_dataset(self, filepath: str):
        """
        Load jsonl pokemon dataset, mask names in bio, and save as csv.

        :param filepath: Filepath of the jsonl pokemon dataset
        """
        # Read json file and mask names in bio
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                record["bio"] = self._mask_name_in_bio(record["bio"], record["name"])
                records.append(record)

        # Store data in class
        df = pd.DataFrame(records, columns=["name", "ndex", "category", "bio", "type1", "type2"])
        self.data = df

        # Get stats on data
        self._compute_stats()
        self._print_summary()

        # Save masked data to csv
        csv_path = os.path.splitext(filepath)[0] + "_masked.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")

    def _mask_name_in_bio(self, bio: str, name: str) -> str:
        """
        Mask the pokemon name in bio.

        :param bio: Pokemon text biology.
        :param name: Pokemon name to mask in biology.
        """
        if not isinstance(bio, str):
            return ""
        pattern = re.compile(re.escape(name), re.IGNORECASE)

        return pattern.sub("[NAME]", bio)

    def _compute_stats(self):
        """
        Get basic stats on dataset.
        """
        if self.data is None:
            raise ValueError("No data loaded")

        bio_lengths = self.data["bio"].apply(lambda x: len(x.split()))
        num_samples = len(self.data)
        num_type1_only = self.data["type2"].isna().sum()
        num_type2 = num_samples - num_type1_only

        self.stats = {
            "num_samples": num_samples,
            "min_bio_len": int(bio_lengths.min()),
            "max_bio_len": int(bio_lengths.max()),
            "mean_bio_len": round(bio_lengths.mean(), 2),
            "median_bio_len": int(bio_lengths.median()),
            "num_type1_only": int(num_type1_only),
            "num_type2": int(num_type2),
        }

    def _print_summary(self):
        """
        Print summary of dataset stats.
        """
        print("----- Dataset Summary -----")
        for k, v in self.stats.items():
            print(f"{k}: {v}")

    def run_classification(self, model_name: str, prompt_type : str):
        # Load model
        self.model = self._load_model(model_name)

        # Check dataset
        if self.data is None:
            raise ValueError("No data loaded. Use load_dataset() first.")

        results = []

        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Classifying Pokemon"):
            description = row["bio"]
            prompt = self._build_prompt(description, prompt_type)
            output_text = self._predict(prompt)
            predicted_types = self._extract_types(output_text)

            # Ensure valid number of types
            if len(predicted_types) == 0:
                predicted_types = [None, None]
            elif len(predicted_types) == 1:
                predicted_types = [predicted_types[0], None]
            elif len(predicted_types) > 2:
                predicted_types = predicted_types[:2]

            # Compute match score
            score = self._compute_match_score(row, predicted_types)

            result_row = {
                "name": row["name"],
                "ndex": row["ndex"],
                "category": row["category"],
                "bio": row["bio"],
                "type1": row["type1"],
                "type2": row["type2"],
                "predicted_type1": predicted_types[0],
                "predicted_type2": predicted_types[1],
                "match_score": score
            }
            results.append(result_row)

        # Save results
        results_df = pd.DataFrame(results)
        csv_path = f"pokemon_llm/outputs/temp_results_{model_name.split('/')[-1]}_{prompt_type}.csv"
        results_df.to_csv(csv_path, index=False)

        # Report accuracy metrics
        mean_score = results_df["match_score"].mean()
        print(f"Average score (0=no match, 0.5=partial, 0.75=wrong order, 1=perfect): {mean_score:.3f}")

        return results_df
    
    def _compute_match_score(self, row, predicted_types):
        """
        Compute order-sensitive match score with categorical grading:
        0.0  = completely_wrong (1 or 2 types)
        0.25 = one_correct_wrong_order (2 types)
        0.5  = one_correct_right_order (2 types)
        0.75 = both_correct_wrong_order (2 types)
        1.0  = completely_correct (1 or 2 types)
        """
        true_types = [t for t in [row.get("type1"), row.get("type2")] if pd.notna(t)]
        preds = [t for t in predicted_types if t is not None]
    
        if not preds or not true_types:
            return 0.0
    
        n_true = len(true_types)
        n_pred = len(preds)
    
        # --- One-type truth cases ---
        if n_true == 1:
            if preds[0] == true_types[0]:
                return 1.0
            elif true_types[0] in preds:
                # predicted it but in wrong slot (if multi-type model output)
                return 0.25
            else:
                return 0.0
    
        # --- Two-type truth cases ---
        if n_true == 2:
            # Case: predicted both
            if n_pred >= 2:
                if preds[:2] == true_types:
                    return 1.0
                elif set(preds[:2]) == set(true_types):
                    return 0.75
                elif any(preds[i] == true_types[i] for i in range(2)):
                    return 0.5
                elif any(pred in true_types for pred in preds[:2]):
                    return 0.25
                else:
                    return 0.0
            # Case: predicted only one
            elif n_pred == 1:
                if preds[0] in true_types:
                    # Got one of them right
                    return 0.5
                else:
                    return 0.0
    
        return 0.0

    def _load_model(self, model_name : str):
        """
        Load the LLM into pipeline.

        :param model_name: The LLM name.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        model.config.pad_token_id = model.config.eos_token_id
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return gen

    def _build_prompt(self, description : str, prompt_type : str):
        """
        Build the prompt for the LLM.

        :param description: The desription of the pokemon to embed in the prompt.
        :param prompt_type: zero_shot, zero_shot_cot, or few_shot.
        """
        base_intro = (
            "You are an expert Pokémon classifier.\n"
            "Your task is to identify the correct Pokémon type(s) from the description below.\n\n"
            "VALID TYPES:\n"
            f"{self.valid_types}\n\n"
        )

        prompt = ""
        if prompt_type == "zero_shot":
            prompt = (
                base_intro +
                "INSTRUCTIONS:\n"
                "1. Read the description carefully.\n"
                "2. Choose ONE type if the Pokémon clearly fits a single type.\n"
                "3. Choose TWO types only if the description strongly mixes two distinct categories.\n"
                "4. NEVER output more than two types.\n"
                "5. Respond ONLY with a single JSON list on the final line.\n"
                "6. Do NOT explain your reasoning or add examples.\n\n"
                "VALID OUTPUT EXAMPLES:\n"
                "[\"Water\"]\n"
                "[\"Ground\", \"Steel\"]\n\n"
                f"DESCRIPTION:\n{description}\n\n"
                "FINAL ANSWER (JSON list only):"
            )

        elif prompt_type == "zero_shot_cot":
            prompt = (
                base_intro +
                "INSTRUCTIONS:\n"
                "1. Read the description and briefly reason about traits, abilities, or habitat.\n"
                "2. Write a short reasoning (1–3 concise sentences).\n"
                "3. Then output ONLY the final answer as a JSON list on the last line.\n"
                "4. Choose ONE type if clear, TWO if both are strongly supported.\n"
                "5. NEVER output more than two types or any text after the list.\n\n"
                "VALID OUTPUT EXAMPLE:\n"
                "Reasoning: Strong muscles and combat focus → Fighting type.\n"
                "Answer: [\"Fighting\"]\n\n"
                f"DESCRIPTION:\n{description}\n\n"
                "Let's think step by step:"
            )

        elif prompt_type == "few_shot":
            prompt = (
                base_intro +
                "INSTRUCTIONS:\n"
                "Use the examples below as references for correct classification.\n"
                "Output only one JSON list on the final line.\n\n"
                "EXAMPLES:\n"
                "Description: [NAME] carries a stick that it uses like a magic wand.\n"
                "Answer: [\"Fire\", \"Psychic\"]\n\n"
                "Description: Its arms are white and rounded, while its feet are dark blue with three toes each.\n"
                "Answer: [\"Water\"]\n\n"
                "Description: [NAME]'s body contains souls burdened with regrets.\n"
                "Answer: [\"Ghost\", \"Flying\"]\n\n"
                f"DESCRIPTION:\n{description}\n\n"
                "FINAL ANSWER (JSON list only):"
            )

        return prompt

    def _predict(self, prompt):
        """
        Run model using prompt.
        
        :param prompt: The prompt.
        """
        output = self.model(prompt, max_new_tokens=50, do_sample=False)

        return output[0]["generated_text"]
    
    def _extract_types(self, text: str):
        """
        Extract valid Pokemon types from model output.
        """
        text = str(text).strip()

        # 1. grab the last JSON-like list in output
        matches = re.findall(r'\[[^\[\]]+\]', text)
        extracted = matches[-1] if matches else ""

        # 2. try to parse it
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, str):
                types = [parsed]
            elif isinstance(parsed, list):
                types = parsed
            else:
                types = []
        except Exception:
            # 3. fallback: look for known types in text
            types = [t for t in self.valid_types if re.search(rf'\b{t}\b', text, re.IGNORECASE)]

        # 4. clean and validate
        clean = []
        for t in types:
            t = str(t).capitalize()
            if t in self.valid_types and t not in clean:
                clean.append(t)
        return clean[:2]

if __name__ == "__main__":
    # Create class
    pokemon_llm = POKEMON_LLM()

    # Load dataset
    pokemon_llm.load_dataset("pokemon_llm/data/pokemon_data_with_pokedex_text.jsonl")

    # Models to test
    very_small_models = ["meta-llama/Llama-3.2-1B-Instruct", "google/gemma-3-1b-it"]
    small_models = ["meta-llama/Llama-3.2-3B-Instruct", "google/gemma-3-4b-it", "Qwen/Qwen3-4B-Instruct-2507"]
    medium_models = ["meta-llama/Llama-3.1-8B-Instruct", "google/gemma-3-12b-it"]
    large_models = ["google/gemma-3-27b-it", "Qwen/Qwen3-30B-A3B-Instruct-2507"]
    very_large_models = ["meta-llama/Llama-3.3-70B-Instruct"]

    # Prompting methods to test
    methods = ["zero_shot", "zero_shot_cot", "few_shot"]

    # Run over models and methods
    tasks = list(itertools.product(very_small_models, methods))
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    model, method = tasks[task_id]

    print(f"=== Running task {task_id}: model={model}, prompt={method} ===")
    pokemon_llm.run_classification(model, method)
