from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
model_name = "Narrativaai/bloom-560m-finetuned-totto-table-to-text"
local_path = os.getenv("HF_HOME", "hf_cache")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_path)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_path)

tokenizer.save_pretrained(local_path)
model.save_pretrained(local_path)



device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

table_text_csv = """"Year","Single","Peak chart positions
AUS","Peak chart positions
AUT","Peak chart positions
BEL
(Fl)","Peak chart positions
BEL
(Wa)","Peak chart positions
FIN","Peak chart positions
FRA","Peak chart positions
GER","Peak chart positions
NED","Peak chart positions
SWE","Peak chart positions
SUI","Certifications
(sales thresholds)","Album"
"2000","\"Around the World (La La La La La)\"","11","1","10","10","7","12","1","5","8","1","GER: Platinum
AUT: Gold
SWI: Gold
FRA: Silver","Planet Pop"
"2000","\"My Heart Beats Like a Drum (Dam Dam Dam)\"","76","6","11","3","12","39","3","37","38","21","GER: Gold","Planet Pop"
"2000","\"Why Oh Why\"","—","16","39","15","—","—","16","—","—","—","","Planet Pop"
"2000","\"Thinking of You\"","—","—","—","—","—","—","46","—","—","51","","Planet Pop"
"2001","\"I'm In Heaven (When You Kiss Me)\"","—","27","—","—","—","—","22","—","—","31","","Touch the Sky"
"2001","\"Call on Me\"","—","—","—","—","—","—","—","—","—","—","","Touch the Sky"
"2001","\"Set Me Free\"","—","—","—","—","—","—","44","—","—","—","","Touch the Sky"
"2001","\"New York City\"","—","—","—","—","—","—","—","—","—","—","","Touch the Sky"

"""

table_text_tsv = """
Year	Single	Peak chart positions\nAUS	Peak chart positions\nAUT	Peak chart positions\nBEL\n(Fl)	Peak chart positions\nBEL\n(Wa)	Peak chart positions\nFIN	Peak chart positions\nFRA	Peak chart positions\nGER	Peak chart positions\nNED	Peak chart positions\nSWE	Peak chart positions\nSUI	Certifications\n(sales thresholds)	Album
2000	"Around the World (La La La La La)"	11	1	10	10	7	12	1	5	8	1	GER: Platinum\nAUT: Gold\nSWI: Gold\nFRA: Silver	Planet Pop
2000	"My Heart Beats Like a Drum (Dam Dam Dam)"	76	6	11	3	12	39	3	37	38	21	GER: Gold	Planet Pop
2000	"Why Oh Why"	—	16	39	15	—	—	16	—	—	—		Planet Pop
2000	"Thinking of You"	—	—	—	—	—	—	46	—	—	51		Planet Pop
2001	"I'm In Heaven (When You Kiss Me)"	—	27	—	—	—	—	22	—	—	31		Touch the Sky
2001	"Call on Me"	—	—	—	—	—	—	—	—	—	—		Touch the Sky
2001	"Set Me Free"	—	—	—	—	—	—	44	—	—	—		Touch the Sky
2001	"New York City"	—	—	—	—	—	—	—	—	—	—		Touch the Sky

"""

table_text_table = """
| Year | Single                                     | Peak chart positions AUS | Peak chart positions AUT | Peak chart positions BEL (Fl) | Peak chart positions BEL (Wa) | Peak chart positions FIN | Peak chart positions FRA | Peak chart positions GER | Peak chart positions NED | Peak chart positions SWE | Peak chart positions SUI | Certifications (sales thresholds)             | Album         |
| 2000 | "Around the World (La La La La La)"        | 11                       | 1                        | 10                            | 10                            | 7                        | 12                       | 1                        | 5                        | 8                        | 1                        | GER: Platinum AUT: Gold SWI: Gold FRA: Silver | Planet Pop    |
| 2000 | "My Heart Beats Like a Drum (Dam Dam Dam)" | 76                       | 6                        | 11                            | 3                             | 12                       | 39                       | 3                        | 37                       | 38                       | 21                       | GER: Gold                                     | Planet Pop    |
| 2000 | "Why Oh Why"                               | —                        | 16                       | 39                            | 15                            | —                        | —                        | 16                       | —                        | —                        | —                        |                                               | Planet Pop    |
| 2000 | "Thinking of You"                          | —                        | —                        | —                             | —                             | —                        | —                        | 46                       | —                        | —                        | 51                       |                                               | Planet Pop    |
| 2001 | "I'm In Heaven (When You Kiss Me)"         | —                        | 27                       | —                             | —                             | —                        | —                        | 22                       | —                        | —                        | 31                       |                                               | Touch the Sky |
| 2001 | "Call on Me"                               | —                        | —                        | —                             | —                             | —                        | —                        | —                        | —                        | —                        | —                        |                                               | Touch the Sky |
| 2001 | "Set Me Free"                              | —                        | —                        | —                             | —                             | —                        | —                        | 44                       | —                        | —                        | —                        |                                               | Touch the Sky |
| 2001 | "New York City"                            | —                        | —                        | —                             | —                             | —                        | —                        | —                        | —                        | —                        | —                        |                                               | Touch the Sky |

"""

tables = {"csv": table_text_csv, "tsv": table_text_tsv, "table": table_text_table}

for t in tables:
    inputs = tokenizer(tables[t], return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(generated_text)
    os.makedirs("totto", exist_ok=True)
    with open(os.path.join("totto", f"gen_text_{t}.txt"), "w", encoding="utf-8") as f:
        f.write(generated_text)
