'''

RUN AS : python3 run_qa.py rep suffixe


Fonction pour run le modele Q&A.
model: modle de Q&A
rep: representation 0, 1, 2 ou 3
data= df qui contient les donnees 
suffixe: "train" or "test"
'''

import numpy as np
import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import argparse
import time

#dictionaire pour mapper plus facilement la representation a la colomne qui va avec
# 0: linéarisation simple, 1: Paragraphe généré par LLM sans Format, 2: Pargraphe genere par LLM avec format
map_rep={
    0: "lin_rep",
    1: "llm_gen",
    2: "llm_format",
    3: "dict_rep",
    5: "json_para",
    999: "baseline"
}


def get_rep_from_table(context, rep, contexts_table):  
    res= contexts_table[contexts_table["context"]==context][map_rep[rep]].values[0] #gets the right representation for the given table "context"
    return res
    
parser = argparse.ArgumentParser()
parser.add_argument("rep", help="Representation number")
parser.add_argument("suffixe", help="Train or test")
args = parser.parse_args()

rep = args.rep
suffixe = args.suffixe
assert suffixe in ["train", "test"], "suffixe is not train or test"

print("Load model ....")
model_name= "deepset/roberta-base-squad2"
qa_model = pipeline("question-answering", model=model_name, tokenizer=model_name)


if suffixe =="train": data= pd.read_csv("WikiTableQuestions-1.0.2-compact/WikiTableQuestions/tagged/data/EN_training_cleaned.tagged")
else: data= pd.read_csv("./WikiTableQuestions-1.0.2-compact/WikiTableQuestions/tagged/data/EN_pristine-unseen-tables_cleaned.tagged")

#verifciation qu'on passe des donnees correctes
assert all(col in data.columns for col in ["context", "utterance"]), "wrong data file passed, doesnt contain contetx or utterance"


table_df= pd.read_csv(f"./WikiTableQuestions-1.0.2-compact/WikiTableQuestions/tables_metadata_{suffixe}.csv") 

print("Run Q&A model ...")  
start= time.time()


#recuper les representations. Si baseline, alor sle context est une dummy value "No context"
contexts=[get_rep_from_table(c, int(rep), table_df) if int(rep)!=999 else "No context." for c in data["context"]] 
results= qa_model(question= data["utterance"].tolist(), context=contexts)
print("Finsihed Running Q&A model")
end= time.time()
#creer nouvelle df pour store les resultats
results_df = data[["id"]].copy()

results_df.loc[:, f"pred_{rep}"]= [r["answer"] for r in results]
results_df.to_csv(f"./WikiTableQuestions-1.0.2-compact/WikiTableQuestions/qa_res_{suffixe}_{rep}.csv",index=False)
results_df= pd.read_csv(f"./WikiTableQuestions-1.0.2-compact/WikiTableQuestions/qa_res_{suffixe}_{rep}.csv")
failed= results_df[f'pred_{rep}'].isna() | (results_df[f'pred_{rep}'] == "")

print(f"The Q&A model failed to generate an answer for: {failed.sum()} instances. Thye were replaced by 'QA Generation Error'")

results_df.loc[failed, f'pred_{rep}'] = 'QA Generation Error'
results_df.to_csv(f"./WikiTableQuestions-1.0.2-compact/WikiTableQuestions/qa_res_{suffixe}_{rep}.csv",index=False)

print(f"Results of Q&A saved to ./WikiTableQuestions-1.0.2-compact/WikiTableQuestions/qa_res_{suffixe}_{rep}.csv")
print(f"Q&A for {suffixe} with rep {rep} done in {(end-start)/60} minutes")
    