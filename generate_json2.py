import pandas as pd
import json
import os
from tqdm import tqdm

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from pathlib import Path



PATH = Path("WikiTableQuestions-1.0.2-compact/WikiTableQuestions")
PATH_TRAIN_CSV = PATH / "tables_metadata_train.csv"
PATH_TEST_CSV = PATH / "tables_metadata_test.csv"

PATH_TEST_JSON = PATH / "test_tables.json"

df_train = pd.read_csv(PATH_TRAIN_CSV)
df_test = pd.read_csv(PATH_TEST_CSV)

#==========================================================CREATION FICHIER JSON======================================================================================
def gen_json():
    file_name_column = pd.concat([df_train["context"],df_test["context"]]).dropna().unique()


    for csv_path in tqdm(file_name_column):
        full_path = PATH / csv_path.replace(".csv", ".tsv")
        if not os.path.exists(full_path):
            print(f"not found : {full_path}")
            continue

        try:
            # df = pd.read_csv(full_path,engine="python",quoting=3,sep=",")
            df = pd.read_csv(full_path, sep="\t")
            print(df)
        except Exception as e:
            print(f"error reading {full_path}: {e}")
            continue

        json_data = df.to_dict(orient="records")
        json_out_path = full_path.with_suffix(".json")

        print(full_path, json_out_path)

        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)


    print("Done.")

    for csv_path in tqdm(file_name_column):
        full_path = PATH / csv_path.replace(".csv", ".json")
        if not os.path.exists(full_path):
            print(f"not found : {full_path}")
            continue

# gen_json()
#==========================================================GENERATION======================================================================================

prompt_base_old= """
You are a helpful assistant. You will read a structured table of data and generate a **single, concise sentence in English** that describes all the information in the table. 
Do not invent facts. Maintain factual correctness. Do not return the table itself.The sentence should include all the information of the provided table. Do not include any instructions or extra commentary. Only write the sentence.

Example:
table: { "Season": "2005–06", "Tier": 3, "Division": "1ª Estatal", "Pos.": "2nd", "Notes": "Promoted" }
sentence: "In the 2005–06 season, the team competed in the third-tier 1ª Estatal division, finished in 2nd place, and earned promotion."

Now do the same for this table:
"""
prompt_base= """
exemple table: { "Season": "2005–06", "Tier": 3, "Division": "1ª Estatal", "Pos.": "2nd", "Notes": "Promoted" }
expected sentence: "In the 2005–06 season, the team competed in the third-tier 1ª Estatal division, finished in 2nd place, and earned promotion."

"""
# device = "cuda:0" 
# cache_dir = os.getenv("HF_HOME")
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# # model_name= "meta-llama/Meta-Llama-3-8B-Instruct"
# # meta-llama/Llama-3.2-3B-Instruct
# tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir = cache_dir)
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir= cache_dir,dtype=torch.float16)
# torch.cuda.empty_cache() 
# model = model.to("cuda")
# print("GPU disponible :", torch.cuda.is_available())
# print("Nombre de GPU :", torch.cuda.device_count())
# print("Nom du GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun")
# print(next(model.parameters()).device)



def generate_paragraph(json_path, max_new_tokens=300):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_paragraphs = []

    for row in data:
        single_prompt = prompt_base+  "table: " + str(row) + "\nsentence (responds one and only sentence):"
        inputs = tokenizer(single_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id= tokenizer.eos_token_id)
        paragraph = tokenizer.decode(outputs[0], skip_special_tokens=True)
        paragraph = paragraph.split("sentence (responds one and only sentence): ")[1]
        paragraph = paragraph.split("\n\n")[0]
        paragraph = str(paragraph).replace("\"", "")

        print("================")
        print(paragraph)
        print("================")
        all_paragraphs.append(paragraph)
        torch.cuda.empty_cache()

    return "\n".join(all_paragraphs)

JSON_FOLDER = PATH / "csv"
OUTPUT_FOLDER = Path("outputs/paragraphs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def gen_paragraph():
    
    for csv_file in tqdm(df_test["context"]):
    # for csv_file in tqdm(reversed(df_test["context"])):
        json_path = PATH / csv_file.replace(".csv", ".json")

        folder = json_path.parent.name
        filename = json_path.stem + ".txt"
        OUTPUT_PATH = OUTPUT_FOLDER / folder / filename

        if os.path.exists(OUTPUT_PATH):
            print(f"fichier existe déjà {OUTPUT_PATH}")
            continue
        print(f"traitement en cours de {json_path}")
        paragraph = generate_paragraph(json_path)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH,"w") as f:
            f.write(paragraph)


# gen_paragraph()


def convert_file():
    
    for csv_file in tqdm(df_test["context"]):
    # for csv_file in tqdm(reversed(df_test["context"])):
        json_path = PATH / csv_file.replace(".csv", ".json")
        print(json_path)
        folder = json_path.parent.name
        filename = json_path.stem + ".txt"
        OUTPUT_PATH = OUTPUT_FOLDER / folder / filename

        target_path = OUTPUT_PATH.with_suffix(".txt")

        os.rename(OUTPUT_PATH,target_path)
# convert_file()

        
def add_rpz_to_table():
    data = []  

    for csv_file in tqdm(df_test["context"]):
        csv_path = Path(csv_file)
        folder = csv_path.parent.name
        filename = csv_path.stem + ".txt"
        path_text = OUTPUT_FOLDER / folder / filename
        with open(path_text, "r") as f:
            content = f.read()

        data.append({
            "context": csv_file,
            "json_para": content
        })

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH / "tmp_json_para.csv", index=False)
    return df_out


# add_rpz_to_table()

df_tmp = pd.read_csv(PATH / "tmp_json_para.csv")
print(f"len df_tmp: {len(df_tmp)}, len df_test: {len(df_test)}")

# Merge json_para into df_test using 'context' as key (preserve all rows from df_test)
df_test = df_test.merge(df_tmp[['context', 'json_para']], on='context', how='left')

missing = df_test['json_para'].isna().sum()
print(f"Added column 'json_para' to df_test. Missing values: {missing}")

df_test.to_csv(PATH_TEST_CSV, index=False)
