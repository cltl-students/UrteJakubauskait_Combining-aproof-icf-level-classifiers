"""
This script processes a dataset of linguistic elements (for example,
'intensifiers.csv', stored in a CSV file and creates an expanded
version of the dataset.

For each row in the original CSV, the original row is kept (with certain
auxiliary columns removed). Additionally, a modified copy of the row is
created using a copy of the text column and a corresponding label column.
The row ID is updated to distinguish it from the original.

The resulting expanded dataset is saved as a new CSV file.
"""

import pandas as pd
import re

df = pd.read_csv("YOUR_INPUT_FILE.csv")

expanded_rows = []

for _, row in df.iterrows():
    original_row = row.drop(["sentence_copy", "new_level"])
    expanded_rows.append(original_row)

    copy_row = row.copy()
    copy_row["text"] = row["sentence_copy"]
    copy_row["original_level"] = row["new_level"]
    copy_row = copy_row.drop(["sentence_copy", "new_level"])
    copy_row["pad_sen_id"] = f"{copy_row['pad_sen_id']}_1"
    expanded_rows.append(copy_row)

expanded_df = pd.DataFrame(expanded_rows)

expanded_df.to_csv(f"YOUR_OUTPUT_FILE.csv", index=False)
