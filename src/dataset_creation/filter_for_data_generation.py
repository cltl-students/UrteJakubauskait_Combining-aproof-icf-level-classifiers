"""
This script prepares my classmate Susana's generated dataset by filtering
sentences to include only those belonging to a predefined set of target
categories, and counts the number of sentences per target category. The filtered
dataset can then be used for level generation.

First, the script loads the input JSON file containing generated labeled
sentences ('train_AMC2023.json'). Then, it filters sentences to keep only those
that contain at least one of the specified target categories (for example,
'B280', 'B134', and 'D760'.). Then, it saves the filtered sentences to a new
JSON file (for example, 'filtered_sentences_new_categories.json'). Then, it
loads an existing filtered file ('filtered_sentences.json') and counts the
number of sentences per target category using a Counter. Finally, it prints the
number of sentences for each target category.
"""


import json
from collections import Counter

input_file = "train_AMC2023.json"
output_file = "filtered_sentences_new_categories.json"

target_categories = ["B280", "B134", "D760", "B164", "D465", 
                     "D410", "B230", "D240"]

with open (input_file, "r", encoding = "utf-8") as f:
    data = json.load(f)

filtered_sentences = []

for entry in data:
    categories = entry.get("categories", [])
    if any(tc in c for c in categories for tc in target_categories):
        filtered_sentences.append(entry)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_sentences, f, ensure_ascii=False, indent=2)

print(f"There are {len(filtered_sentences)} sentences in the filtered file.")

with open("filtered_sentences.json", "r", encoding="utf-8") as f:
    filtered_data = json.load(f)

category_counter = Counter()

for entry in filtered_data:
    categories = entry.get("categories", [])
    for cat in categories:
        for code in target_categories:
            if code in cat:
                category_counter[code] += 1

for code in target_categories:
    print(f"{code}: {category_counter[code]} sentences.")
