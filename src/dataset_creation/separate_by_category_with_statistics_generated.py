"""
This script processes OpenAI-generated labeled sentences by combining multiple 
JSON files, ensuring uniqueness, splitting entries by category, and generating 
summary statistics.

First, it loads sentences from multiple JSON files. Then, it
combines sentences while ensuring uniqueness based on the tuple ('note_id' and
'sentence_index'). Then, it checks that the number of categories matches the
number of labels for each sentence, reporting any mismatches. For each
category-label pair, the script standardizes category names starting with
'B240' to 'D240'. Then, it creates a new entry containing only one category and
its corresponding label. Finallt, it stores entries in a dictionary keyed by
category. After this, the script saves entries for each category into separate
JSON files in the 'files_by_category' folder, effectively splitting the original
OpenAI-generated file by category. Then, it calculates sentence length
statistics for each category: number of entries, average sentence length (words),
median sentence length (words), and shortest and longest sentence (words).
Finally, it writes these statistics to a text file.
"""

import json
from pathlib import Path 
import statistics
import copy
import pandas as pd

files = ["YOUR_INPUT_FILE_1.json", "YOUR_INPUT_FILE_2.json"]
combined_data = []
seen_sentences = set()

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for sentence in data:
            key = (sentence["note_id"], sentence["sentence_index"])
            if key not in seen_sentences:
                combined_data.append(sentence)
                seen_sentences.add(key)

print(f"Total unique sentences after combining: {len(combined_data)}")

output_folder = Path("files_by_category")
output_folder.mkdir(exist_ok=True)

category_dict = {}
mismatch_count = 0

for entry in combined_data:
    categories = entry["categories"]
    labels = entry["labels"]
    
    if len(categories) != len(labels):
        mismatch_count += 1
        print(f" Mismatch found: {entry['sentence']}, {categories}, {labels}")
        continue

    for category, label in zip(categories, labels):
        if category.strip().startswith("B240"):
            category = "D240" + category[4:]

        new_entry = copy.deepcopy(entry)
        new_entry["categories"] = category
        new_entry["labels"] = label

        category_dict.setdefault(category, []).append(new_entry)

with open('YOUR_OUTPUT_FILE.txt', "w", encoding="utf-8") as f_out:
    f_out.write("Statistics for OpenAI generated files\n")
    f_out.write("=" * 70 + "\n\n")

    for category, entries in category_dict.items():
        safe_category = category.replace(" ", "_").replace("/", "-")
        output_file = output_folder / f"{safe_category}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

        lengths = [len(entry["sentence"].split()) for entry in entries]

        avg_len = sum(lengths) / len(lengths)
        median_len = statistics.median(lengths)
        min_len = min(lengths)
        max_len = max(lengths)

        f_out.write(f"Category: {category}\n")
        f_out.write(f"Number of rows: {len(entries)}\n")
        f_out.write(f"Average length: {avg_len:.2f}\n")
        f_out.write(f"Median length: {median_len:.2f}\n")
        f_out.write(f"Shortest sentence (words): {min_len}\n")
        f_out.write(f"Longest sentence (words): {max_len}\n")
        f_out.write("-" * 70 + "\n\n")

print(f"Number of mismatched entries (categories vs. labels): {mismatch_count}")
