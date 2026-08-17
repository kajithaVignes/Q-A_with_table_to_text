'''
RUN WITH COMMAND: python3 eval_stat.py suffixe rep
like : python3 eval_stat.py 0 train


File to evaluate the EM and F1 score for every question. Used for statistical tests.
Run with command: python3 eval.py rep suffixe
rep is la representation 
suffixe is "train" or "test"
'''

#rep 0 for linerization, 1 for llm ...etc voir le dictionaire map_rep
#RUN FILE WITH 1, 2, 3 OR 4 AS FILE NAMES 
#read file name from comd line

import numpy as np
import argparse
import time
import csv
import pandas as pd
from evaluate import load

parser = argparse.ArgumentParser()
parser.add_argument("rep", help="Representation number")
parser.add_argument("suffixe", help="Train or test")
args = parser.parse_args()

rep = int(args.rep)
suffixe = args.suffixe

assert rep in [0,1,2,3, 999], "Rep passed is wrong, should be 0, 1,2 , 3 or 999"
assert suffixe in ["train","test"], "Suffixe passed is wrong, should be 'train' or 'test'"

print(f"Currently Running evaluation for Rep {rep} on {suffixe}")


res_data= pd.read_csv(f"./WikiTableQuestions-1.0.2-compact/WikiTableQuestions/qa_res_{suffixe}_{rep}.csv")
if suffixe=="train":
    data= pd.read_csv("WikiTableQuestions-1.0.2-compact/WikiTableQuestions/tagged/data/EN_training_cleaned.tagged")
else:
    data= pd.read_csv("WikiTableQuestions-1.0.2-compact/WikiTableQuestions/tagged/data/EN_pristine-unseen-tables_cleaned.tagged")


data_and_res= pd.merge(res_data, data[["id","targetValue"]], on="id", how="inner") #merge both files to access predictions and targetValue


metric = load("squad")
pred_formatted = [{"id": str(i), "prediction_text": p} for i, p in enumerate(data_and_res[f'pred_{rep}'].tolist())]
target_formatted  = [{"id": str(i), "answers": {"text": [t], "answer_start": [0]}}  # answer_start ignored
         for i, t in enumerate(data_and_res['targetValue'].tolist())]

print("Start score comupting ...")
start= time.time()
i=0
scores=[]
for pred, ref in zip(pred_formatted, target_formatted):
    if i%100==0: print(i)
    s=metric.compute(predictions=[pred], references=[ref])
    scores.append(s)
    i+=1

end= time.time()

print(" finished score comupting")

data_and_res[f"F1_stat_score_pred_{rep}"] = [s["f1"] for s in scores]
data_and_res[f"EM_stat_score_pred_{rep}"] = [s["exact_match"] for s in scores]
data_and_res.drop(columns=["targetValue"], inplace=True)
data_and_res.to_csv(f"./WikiTableQuestions-1.0.2-compact/WikiTableQuestions/qa_res_{suffixe}_{rep}.csv",index=False)
print(f"Evaluation saved to ./WikiTableQuestions-1.0.2-compact/WikiTableQuestions/qa_res_{suffixe}_{rep}.csv")

em_score= data_and_res[f"EM_stat_score_pred_{rep}"].mean()
f1_score= data_and_res[f"F1_stat_score_pred_{rep}"].mean()

print(f"Finished run in {(end-start) /60} minutes.")
print('exact_match', em_score, 'f1', f1_score)