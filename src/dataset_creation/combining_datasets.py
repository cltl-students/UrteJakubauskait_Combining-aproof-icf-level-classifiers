"""
This script combines existing training and development datasets (stored as 
pickle files) with additional (generated) entries from a JSON file, and saves
the combined datasets as new pickle files.

Firstly, it loads the existing training and development datasets from pickle
files (for example, 'train.pkl' and 'dev.pkl') and converts them to record
dictionaries. Then, it reads a JSON file containing additional entries to be
added. If an entry contains a 'sentence' field, it is renamed to 'text'. Then,
the script shuffles the new entries and splits them into 90% for training and
10% for development. Then, it extends the existing training and development
datasets with the new entries. Afterwards, it shuffles the combined datasets
and saves the combined datasets as 'train_combined.pkl' and 'dev_combined.pkl'.
Finally, it prints the total number of entries in the combined training and
development sets.
"""

import json
from pathlib import Path
import random
import pickle
import pandas as pd

folder = Path(".")

train_file = "YOUR_TRAIN_FILE.pkl"
dev_file =  "YOUR_DEV_FILE.pkl"

train_data = []
dev_data = []   

train_data = pd.read_pickle(train_file).to_dict(orient="records")
dev_data = pd.read_pickle(dev_file).to_dict(orient="records")

train_combined = []
dev_combined = []

train_combined.extend(train_data)
dev_combined.extend(dev_data)

json_file = next(folder.glob("*.json"))
with open(json_file, "r", encoding="utf-8") as f:
    new_entries = json.load(f)

for entry in new_entries:
    if "sentence" in entry:
        entry["text"] = entry.pop("sentence")

random.shuffle(new_entries)
split_idx = int(0.9 * len(new_entries))
train_combined.extend(new_entries[:split_idx])
dev_combined.extend(new_entries[split_idx:])

random.shuffle(train_combined)
random.shuffle(dev_combined)

with open("train_combined.pkl", "wb") as f:
    pickle.dump(train_combined, f)

with open("dev_combined.pkl", "wb") as f:
    pickle.dump(dev_combined, f) 

print(f"Total train entries: {len(train_combined)}")
print(f"Total dev entries: {len(dev_combined)}")
