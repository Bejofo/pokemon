import requests
import time
import pandas as pd

def scrape(path_chunk):
    ans = dict()
    for i in range(1,999):
        time.sleep(0.2)
        url = f"https://pokeapi.co/api/v2/{path_chunk}/{i}/"
        r = requests.get(url)
        
        if r.status_code == 404:
            break
        
        group_name = r.json().get("name")
        appliciable_pokemons = r.json().get("pokemon_species")
        for poke in appliciable_pokemons:
            ans[poke.get("name").lower()] = group_name
        print(group_name,len(appliciable_pokemons))
    return ans

def scrape_all_egg_groups():
    return scrape("egg-group")

def scrape_all_shapes():
    return scrape("pokemon-shape")

def scrape_all_color():
    return scrape("pokemon-color")

def scrape_all_habitat():
    return scrape("pokemon-habitat")



def add_col(pokedata,col_name,scraped_data):
    def retrive(row):
        pokemon_name = row['english_name'].lower()
        if pokemon_name in scraped_data:
            return scraped_data[pokemon_name]
        else:
            return pd.NA
    
    pokedata[col_name] = pokedata.apply(retrive, axis=1)


import pandas as pd

p = pd.read_csv('/media/sf_Downloads/pokemon.csv', encoding='utf-16-le', sep='\t')
p = p[[
    'national_number','gen','english_name','japanese_name','classification',
    'primary_type','secondary_type','description',
]]

add_col(p,"egg_group", scrape_all_egg_groups())
add_col(p,"color", scrape_all_color())
add_col(p,"habitat", scrape_all_habitat())
add_col(p,"shape", scrape_all_shapes())


# p.save()
p.to_csv('augmented_data.csv')

import code
code.interact(local=locals())