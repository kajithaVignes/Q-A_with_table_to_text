
#FUNCTION TO FLATTN THE TABLE INTO A DISCTIONARY-LIKE REPRESENTATION
"""
Example:
c1 | c2|
10 | 20|
100 | 200|

Dictionary Rep:

' Row 0 : {"c1": 10, "c2": 20},
  Row 1:  {"c1":100, "c2": 200}  '
"""
#CHANGE PATH ACCORDINGLY


import csv
import sys
import pandas as pd

csv.field_size_limit(sys.maxsize) 

file_name="./WikiTableQuestions-1.0.2-compact/WikiTableQuestions/tables_metadata_train.csv"
all_tables_df= pd.read_csv(file_name)
dictionary_rep=[]
for idx,row in all_tables_df.iterrows():
  
    try:
        table= pd.read_csv(f"WikiTableQuestions-1.0.2-compact/WikiTableQuestions/{row['context']}")
        
    except Exception as e:
        try:
            # print("Failed , parser")
            table= pd.read_csv(f"WikiTableQuestions-1.0.2-compact/WikiTableQuestions/{row['context'][:-4]}.table", sep="|", engine="python")
            
            
        except Exception as e2:
            X = []
            print("Failed | parser")
            with open(f"WikiTableQuestions-1.0.2-compact/WikiTableQuestions/{row['context']}", newline='', encoding="utf-8") as f:
                
                reader = csv.reader(f, quotechar='"', escapechar='\\')
                for x in reader:
                    X.append(x)

                table = pd.DataFrame(X[1:], columns=X[0])

    
    
    rep=""
    for idxi,rowi in table.iterrows():
        row_str = ",".join(f"{str(c).strip()}: {str(v).strip()}" for c, v in rowi.items())
        rep+=f"Row {idxi}:{{{ row_str }}}\n"
        


    dictionary_rep.append(rep)


all_tables_df["dict_rep"]= dictionary_rep
all_tables_df= all_tables_df.to_csv(file_name, index=False)


        
        
        
    