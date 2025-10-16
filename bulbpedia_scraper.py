import requests
import json
from bs4 import BeautifulSoup
import time
import os 
# import regex as re

# a = Array.from(document.querySelectorAll("a")).map(x=>x.href).filter(x=> x.includes("/wiki/") && x.includes("_(Pok%C3%A9mon)"))
# b = new Set(a)
# JSON.stringify(Array.from(a))

def scrape_page(url):
    if "?&action=edit" not in url:
        url += "?&action=edit"

    r = requests.get(url)
    
    soup = BeautifulSoup(r.content)
    return soup.select_one('#wpTextbox1').text


with open("all_bulpedia_pokemons.json") as f:
    j = json.load(f)
    
for url in j:
    filename = url.split("/wiki/")[-1]
    filename = filename.rsplit("_",1)[0] + ".txt"
    rel_filepath = f"./bulbapedia_src/{filename}"
    if os.path.exists(rel_filepath):
        print(f"skipping {filename}")
        continue
    
    src_text = scrape_page(url)
    with open(rel_filepath,"w") as f:
        f.write(src_text)
    print(filename)
    
    time.sleep(0.42)