import json

with open("pokemon_llm/data/pokemon_data.jsonl", "r", encoding="utf-8") as f:
    raw = f.read()

entries = raw.strip().split("\n}\n{")
entries = ["{" + e.strip().strip("{}") + "}" for e in entries]

with open("pokemon_llm/data/pokemon_data_fixed.jsonl", "w", encoding="utf-8") as out:
    for e in entries:
        try:
            obj = json.loads(e)
            out.write(json.dumps(obj) + "\n")
        except json.JSONDecodeError:
            continue
