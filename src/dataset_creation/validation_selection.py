"""
This script selects a small random subset of examples per category from labeled
JSON files and saves the selected examples as an Excel file.

Firstly, it reads all input JSON files (for example, "labeled_sentences.json" 
and "labeled_sentences_new_categories.json") and combines the entries. Then, 
it groups the examples by their categories. For each category, it randomly 
shuffles the examples and selects up to 5 examples. The script adds a 
"selected_category" field indicating the category for which each example was 
selected. Finally, it saves the selected examples to an Excel file 
(for example, "validation_selection.xlsx").
"""

import json
import random
from collections import defaultdict
import pandas as pd

input_files = ["labeled_sentences.json", "labeled_sentences_new_categories.json"]
output_file = "validation_selection.xlsx"

data = []
for file in input_files:
    with open(file, encoding="utf-8") as f:
        data.extend(json.load(f))

examples_by_cat = defaultdict(list)
for entry in data:
    categories = entry.get("categories", [])
    for cat in categories:
        examples_by_cat[cat].append(entry)

rows = []
for cat, entries in examples_by_cat.items():
    random.shuffle(entries)
    for entry in entries[:5]:
        row = entry.copy()
        row["selected_category"] = cat
        rows.append(row)

df = pd.DataFrame(rows)
df.to_excel(output_file, index=False)
