# MEDS

Scripts et analyses pour la generation de representations textuelles de tables WikiTableQuestions, puis l'evaluation d'un modele de question-reponse.

## Contenu

- `generate_table_rep_dictionary.py` : representation dictionnaire d'une table.
- `generate_llm_rep_1.py` : generation d'un paragraphe, ligne par ligne.
- `generate_llm_rep_format_2.py` : generation guidee, une phrase par ligne.
- `generate_json.py` et `generate_json2.py` : conversion des tables et generation de paragraphes.
- `run_qa.py` : execution du modele de question-reponse.
- `eval_stat.py` : calcul des scores EM et F1.
- `notebooks` : analyses exploratoires et visualisations conservees a la racine pour compatibilite avec les chemins historiques.
- `fig/` : figures produites par les analyses.

Les donnees WikiTableQuestions et les resultats de generation sont volontairement ignores par Git. Consulte la documentation du dataset dans `WikiTableQuestions-1.0.2-compact/WikiTableQuestions/README.md` pour obtenir les donnees necessaires.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Les modeles Hugging Face peuvent necessiter une authentification et l'acceptation de leurs conditions d'utilisation. Configure ton token localement, sans le mettre dans le code :

```bash
export HF_TOKEN="hf_..."
huggingface-cli login
```

Le token n'est pas obligatoire pour les scripts qui utilisent uniquement des modeles publics sans restriction.

## Utilisation

Les commandes doivent etre lancees depuis la racine du projet, une fois les fichiers du dataset disponibles :

```bash
python run_qa.py 0 test
python eval_stat.py 0 test
python generate_llm_rep_1.py test 1
python generate_llm_rep_format_2.py test 1
```

Les scripts de generation de modeles peuvent necessiter un GPU et plusieurs gigaoctets de memoire. Les fichiers produits sont ecrits dans les donnees locales ou dans `outputs/`, et ne doivent pas etre committes.

## Donnees et reproductibilite

Le projet utilise WikiTableQuestions, decrit dans `WikiTableQuestions-1.0.2-compact/WikiTableQuestions/README.md`. Les donnees, caches de modeles, predictions et sorties intermediaires ne sont pas inclus dans ce depot afin de garder le depot leger et de respecter les conditions de redistribution du dataset.

