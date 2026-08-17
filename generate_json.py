import pandas as pd
import json
import os
from tqdm import tqdm

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
        except Exception as e:
            print(f"error reading {full_path}: {e}")
            continue

        json_data = df.to_dict(orient="records")
        json_out_path = full_path.with_suffix(".json")


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
prompt_base = "You are a helpful assistant. You will read a structured table of data and generate a **concise, coherent paragraph in English**. The paragraph should include all relevant information in a readable way, maintaining factual correctness. Do not invent facts. Format the paragraph in full sentences, suitable for Q&A. You are a helpful assistant. You will read a structured table of data and generate a **concise, coherent paragraph in English**. The paragraph should include all relevant information in a readable way, maintaining factual correctness. Do not invent facts. Format the paragraph in full sentences, suitable for Q&A. "

if os.getenv("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

device = "cuda:0" 
model_name= "meta-llama/Meta-Llama-3-8B-Instruct"
cache_dir = os.getenv("HF_HOME")
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir = cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir= cache_dir)
model.to(device)


def generate_paragraph(json_path, max_new_tokens=300):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # prompt = "You are a helpful assistant. You will read a structured table of data and generate a concise, coherent paragraph in English. Include all relevant information in readable sentences, suitable for Q&A. Do not invent facts.\n\n"
    prompt = prompt_base
    for row in data:
        prompt += " | ".join(f"{k}: {v}" for k, v in row.items()) + "\n"

    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    paragraph = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paragraph



JSON_FOLDER = PATH / "csv"


def gen_paragraph_from_json():
    i = 5
    for folder_name in tqdm(os.listdir(JSON_FOLDER)):
        folder_name = JSON_FOLDER+"/"+folder_name
        for file_name in tqdm(os.listdir(folder_name)):
            if not file_name.endswith(".json"):
                continue

            json_path = os.path.join(folder_name, file_name)
            paragraph = generate_paragraph(json_path)
            print(paragraph)
            
