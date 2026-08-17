import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import argparse
import time
import os

"""
Script to generate the LLM representation that is a row by row summarization (LLM1 simple).
RUN FILE WITH  : python3 generate_llm_rep_1.py suffixe file_number |  file number is 1,2,3, ou 4.  
REPRESENTATION EST FIXE ICI (REP=1)
suffixe is train or test
"""

#read file name from comd line

parser = argparse.ArgumentParser()
parser.add_argument("suffixe", help="Train or test")
parser.add_argument("filename", help="File part number")
args = parser.parse_args()

filename = int(args.filename)
suffixe = args.suffixe

assert filename in [1,2,3,4], "file name passed is wrong, should be 1,2 , 3or 4"
assert suffixe in ["train","test"], "Suffixe passed is wrong, should be 'train' or 'test'"

#erad table
table_df= pd.read_csv(f"tables_metadata_{suffixe}_part{filename}.csv")
print(f"\nRead table 'tables_metadata_{suffixe}_part{filename}.csv' , having {len(table_df)}  lines\n")

#connext to hugging face
if os.getenv("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

#laod LLM 
model_name= "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=600,
    temperature=0.7
)

print(f"\nModel {model_name} loaded and ready for use\n")

#set rpompt

prompt= """You are an expert data analyst that converts tables into text.
For a given table, generate a single paragraph summarizing the table row by row, using the values of each column. 
You must include all information and values present in the table in your paragrph. The first row represents column names.
In your answer, include only the paragraph and nothing additional. Be neutral and precise. 
Example: \n Table:\n {table_example} \n Paragraph: {exp_example} \n 
Now, do the same for this table. \n Table:\n {table} \n Paragraph:  """

table_example="| Party      | Active Voters | Inactive Voters | Total Voters | Percentage |\n| Democratic | 3,683         | 251             | 3,934        | 36.93%     |\n"
exp_example="The Democratic Party has 3,683 active voters and 251 inactive voters, for a total of 3,934 voters, which represents 36.93% of all voters."


prompts= [prompt.format(table=t, table_example= table_example, exp_example=exp_example) for t in table_df["lin_rep"] ]

print(f"\nPrompts formatted, nb of prompts: {len(prompts) }\n")

#generate

print(f"Generating representation ....")
st= time.time()
res= generator(prompts)
end= time.time()
print(f"Finished Generation")
print(f"Took {(end-st)/60} minutes to generate {len(prompts)} prompts.")
#save to file

print("Saving models")
table_df["res_llm_gen"]= res
table_df.to_csv(f"./WikiTableQuestions-1.0.2-compact/WikiTableQuestions/tables_metadata_{suffixe}_part{filename}.csv", index=False)
llm_rep= [r[0]['generated_text'].split("Paragraph:")[-1] for r in res]
table_df["llm_gen"]= llm_rep
table_df["llm_gen"] = table_df["llm_gen"].fillna("LLM generation error")
nb_fail= len( table_df[table_df["llm_gen"]=="LLM generation error"])
print(f"Number of times LLm failed to generate= {nb_fail} ")
print(f"Percentage of times LLm failed to generate for rep1, for file {filename} in {suffixe}= {nb_fail*100/ len(table_df)} ")


table_df.to_csv(f"./WikiTableQuestions-1.0.2-compact/WikiTableQuestions/tables_metadata_{suffixe}_part{filename}.csv", index=False)
print("Models saved hallelujah!")

