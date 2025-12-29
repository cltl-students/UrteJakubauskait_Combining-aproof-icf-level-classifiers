"""
This script selects a small random sample of sentences containing specific 
intensifier words from multiple filtered training datasets and combines them 
into a single CSV file.

Firstly, it iterates over all CSV files in the specified input folder. For each
file, it reads the dataset into a DataFrame. Then, it filters rows where the
'text' column contains any of the predefined intensifier words. Then, randomly
selects up to 'sample_size' rows from the filtered subset and creates a copy of
the selected sentences as 'sentence_copy'. Afterwards, the script renames
'labels' column to 'original_level' and adds a 'new_level' column. Then, it
inserts the new columns next to the 'text' column for consistency and adds a
'source_file' column indicating the origin file. Then, it combines all selected
rows into a single DataFrame. Finally, it saves the combined DataFrame as
'selected_intensifiers_sentences.csv'.

Parameters:
- input_folder: folder containing filtered CSV datasets.
- output_csv: name of the resulting combined CSV file.
- sample_size: maximum number of rows to select per file.
"""

import os
import random
import pickle
import pandas as pd 

input_folder = "YOUR_INPUT_FOLDER"
output_csv = "selected_intensifiers_sentences.csv"
sample_size = 6

intensifiers_words = ["enorm", "enorme", "gigantisch", 
                      "gigantische", "reusachtig", "reusachtige","kolossaal", "kolossale", 
                      "ontzettend", "ontzettende", "erg", "erge", "heel", "zeer", "vreselijk", 
                      "vreselijke", "flink", "flinke", "ernstig", "ernstige", "zwaar", "heftig", 
                      "heftige", "ingrijpend", "ingrijpende", "behoorlijk", "behoorlijke", 
                      "stevig", "stevige", "krachtig", "krachtige", "heel wat", "flink wat", 
                      "behoorlijk wat", "een boel", "veel", "in hoge mate", "aanzienlijk"]

all_selected = []

for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder, filename)

    df = pd.read_csv(filepath)

    mask = df["text"].astype(str).str.lower().apply(lambda s: any(inf in s for inf in intensifiers_words))
    inf_df = df[mask]

    selected_rows = inf_df.sample(n=min(sample_size, len(inf_df)), random_state=42)
    selected_rows = selected_rows.copy()
    selected_rows["source_file"] = filename

    selected_rows["sentence_copy"] = selected_rows["text"]
    selected_rows.rename(columns={"labels": "original_level"}, inplace=True)
    selected_rows["new_level"] = ""

    cols = list(selected_rows.columns)
    for c in ["sentence_copy", "original_level", "new_level"]:
        if c in cols:
            cols.remove(c)
    
    text_idx = cols.index("text") + 1
    for i, c in enumerate(["sentence_copy", "original_level", "new_level"]):
        cols.insert(text_idx + i, c)

    selected_rows = selected_rows[cols]
    all_selected.append(selected_rows)

combined_df = pd.concat(all_selected, ignore_index=True)
combined_df.to_csv(output_csv, index=False, encoding="utf-8")
