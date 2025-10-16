import re
import json
import glob
# All files and directories ending with .txt and that don't begin with a dot:

all_bios = []


def get_categorical(contents):
    ans = {}
    t = re.search(r"{{Pokémon Infobox.*?}}", contents, re.DOTALL | re.MULTILINE)
    for match in re.finditer(r"^\|.*?=.*?$",t.group(),flags= re.MULTILINE):
        tup = match.group()[1:].split("=")
        ans[tup[0]] = tup[1]
    return ans 
        


def process_biology(contents):
    regex = r"==Biology==.*?\n=.*?=\n"
    matches = re.search(regex, contents, re.DOTALL | re.MULTILINE)
    m = matches.group()
    m = m.rsplit("\n",2)[0]
    m = m.strip()
    m = m.replace('{{OBP|Pokémon|species}}',"Pokémon")
    m = m.replace('{{Spoilers}}',"")
    m = m.replace('{{endspoilers}}',"")
    m = m.replace('{{Endspoilers}}',"")
    m = m.replace('{{left clear}}',"")

    # todo remove href and gallery    
    # fix Koraidon

    m = re.sub(r"<gallery.*?>.*?</gallery>","",m, flags=re.DOTALL | re.MULTILINE)
    m = re.sub(r"<ref>.*?</ref>","",m, flags=re.DOTALL | re.MULTILINE)
    # m = re.sub(r"<ref name.*?</ref>","",m, flags=re.DOTALL | re.MULTILINE)



    bad_prefixes = ["[[File:","File:"]

    lines = [l for l in m.splitlines() if ( not any(l.strip().startswith(p) for p in bad_prefixes) ) ]
    m = "\n".join(lines)

    tags_regex = r"(?<={{).*?(?=}})|(?<=\[\[).*?(?=\]\])" 

    new_m = m
    replacements = {}

    for match in re.finditer(tags_regex,m):
        matched = match.group()
        if "|" in matched:
            matched = matched.split("|")[-1]
        replacements[match.span()] = matched

    for span,r in sorted(replacements.items(),reverse=True):
        new_m = new_m[:span[0]-2] + r + new_m[span[1]+2:]   
    return new_m

for filename in glob.glob("./bulbapedia_src/*.txt"):
    with open(filename,"r") as f:
        contents = f.read()
        # bio = process_biology(contents)
        ans = get_categorical(contents)
        ans["bio"] = process_biology(contents)
        print(json.dumps(ans,indent=2),"\n\n")
        

# biology_seciton_regex = r"==Biology==.*?\n=.*?=\n"gms

# for 