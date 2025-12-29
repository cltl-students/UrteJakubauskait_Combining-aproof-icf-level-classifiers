"""
This script updates a training dataset pickle file by removing sentences 
that have been selected for testing in error analysis.

Firstly, it loads the training dataset from a pickle file. Then, it reads 
multiple CSV files containing selected sentences (for example, minimizers, 
intensifiers, and negations). Then, it filters each CSV to include only rows 
from a specific source file and collects the corresponding sentence IDs. Then,
it removes all sentences from the pickle dataset that have matching IDs in the
collected set and saves the filtered dataset back to the same pickle file.
Finally, the script prints statistics about the original dataset, number of
sentences removed, and new dataset size.

Parameters:
- pkl_file: path to the training dataset pickle file.
- csv_files: list of CSV files containing selected sentences to remove.
"""

import pickle
import pandas as pd

pkl_file = "YOUR_INPUT_FILE.pkl"
csv_files = ["selected_minimizers_sentences_pv.csv",
              "selected_intensifiers_sentences_pv.csv",
              "selected_negation_sentences_pv.csv"]
              
data = pd.read_pickle(pkl_file)

print(type(data))
print(len(data))

sent_ids_to_remove = set()

for file in csv_files:
    df = pd.read_csv(file)
    df_filtered = df[df["source_file"] == "YOUR_OUTPUT_FILE.csv"]
    sent_ids_to_remove.update(df_filtered["pad_sen_id"])

print(f"Total number of sentences to be removed: {len(sent_ids_to_remove)}")

filtered_data = data.loc[data["pad_sen_id"].isin(sent_ids_to_remove) == False]

with open(pkl_file, "wb") as f:
    pickle.dump(filtered_data, f)

print("Original sentence count:", len(data))
print("New sentence count:", len(filtered_data))
print("Removed:", len(data) - len(filtered_data))
