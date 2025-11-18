import pandas as pd
import json
import re
from tqdm import tqdm
import os
import itertools
import textdistance
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
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = []
        for record in data:
            record["bio"] = self._mask_name_in_bio(record.get("bio", ""), record.get("name", ""))
            records.append(record)

        df = pd.DataFrame(records, columns=["name", "ndex", "category", "bio", "type1", "type2"])
        self.data = df

        self._compute_stats()
        self._print_summary()

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

            if not isinstance(predicted_types, list):
                predicted_types = []

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
                "match_score": score,
                "raw_output": output_text
            }
            results.append(result_row)

        # Save results
        results_df = pd.DataFrame(results)
        csv_path = f"pokemon_llm/outputs/results_{model_name.split('/')[-1]}_{prompt_type}.csv"
        results_df.to_csv(csv_path, index=False)
    
    def _compute_match_score(self, row, predicted_types):
        if not isinstance(predicted_types, list):
            predicted_types = []
        predicted_types = [t for t in predicted_types if isinstance(t, str)]

        true_types = [t for t in [row.get("type1"), row.get("type2")] if isinstance(t, str)]

        # if either list empty, similarity = 0
        if not predicted_types or not true_types:
            return 0.0

        return textdistance.damerau_levenshtein.normalized_similarity(predicted_types, true_types)

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
                "1. Read the DESCRIPTION carefully.\n"
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
                "1) Read the description and provide a brief reasoning only.\n"
                "2) Immediately after that reasoning, on the NEXT LINE, output ONLY the final answer as a JSON list.\n"
                "3) The JSON list must contain AT MOST two string items drawn from VALID TYPES and nothing else.\n"
                "4) Do NOT include any other text, labels, or punctuation after the JSON list.\n\n"
                "FORMAT (exactly):\n"
                "Reasoning: <brief sentence(s)>\n"
                "[\"Type1\"]  OR  [\"Type1\", \"Type2\"]\n\n"
                f"DESCRIPTION:\n{description}\n\n"
                "Produce reasoning (brief sentence(s)) and then the JSON list on the next line.\n"
                "Let\'s think step by step..."
            )

        elif prompt_type == "few_shot":
            prompt = (
                base_intro +
                "INSTRUCTIONS:\n"
                "1. Read the DESCRIPTION carefully.\n"
                "2. The EXAMPLES show correct answers only; do NOT imitate their structure or produce objects.\n"
                "3. Choose ONE type if the Pokémon clearly fits a single type.\n"
                "4. Choose TWO types only if the description strongly mixes two distinct categories.\n"
                "5. NEVER output more than two types.\n"
                "6. Respond ONLY with a single JSON list on the final line.\n"
                "7. Do NOT explain your reasoning or add examples.\n\n"
                "EXAMPLES:\n"
                "Description: [NAME] carries a stick that it uses like a magic wand.\n"
                "Answer: [\"Fire\", \"Psychic\"]\n\n"
                "Description: Its arms are white and rounded, while its feet are dark blue with three toes each.\n"
                "Answer: [\"Water\"]\n\n"
                "Description: [NAME]'s body contains souls burdened with regrets.\n"
                "Answer: [\"Ghost\", \"Flying\"]\n\n"
                f"DESCRIPTION:\n{description}\n\n"
                "FINAL ANSWER (JSON list):"
            )

        return prompt

    def _predict(self, prompt):
        """
        Run model using prompt.
        
        :param prompt: The prompt.
        """
        try:
            outputs = self.model(prompt, max_new_tokens=200, do_sample=False, return_full_text=False)
            if not outputs or "generated_text" not in outputs[0]:
                return ""
            return outputs[0]["generated_text"]
        except Exception:
            return ""
    
    def _extract_types(self, text: str):
        text = str(text)

        match = re.search(r'\[(?=[^\]]*")(.*?)\]', text)
        if not match:
            return []   # return empty list (never None)

        candidate = "[" + match.group(1) + "]"

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
            else:
                return []
        except Exception:
            return []

if __name__ == "__main__":
    # Create class
    pokemon_llm = POKEMON_LLM()

    # Load dataset
    pokemon_llm.load_dataset("pokemon_llm/data/pokemon_data.json")

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
