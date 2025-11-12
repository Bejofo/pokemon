import json

# Input file paths
pokemon_data_file = "pokemon_llm/data/pokemon_data_fixed.jsonl"
pokedex_file = "pokemon_llm/data/pokedex_newest.json"
output_file = "pokemon_llm/data/pokemon_data_with_pokedex_text.jsonl"

# Load pokedex data (list of dicts)
with open(pokedex_file, "r", encoding="utf-8") as f:
    pokedex_data = json.load(f)

# Create lookup table: pokedex_num -> text
pokedex_lookup = {entry["pokedex_num"]: entry["text"] for entry in pokedex_data}

# Process JSONL file and replace bio
with open(pokemon_data_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:
    
    for line in infile:
        if not line.strip():
            continue  # skip empty lines
        pokemon = json.loads(line)
        ndex = pokemon.get("ndex")
        
        if ndex in pokedex_lookup:
            pokemon["bio"] = pokedex_lookup[ndex]
        
        json.dump(pokemon, outfile, ensure_ascii=False)
        outfile.write("\n")

print(f"Done. Output written to {output_file}")
