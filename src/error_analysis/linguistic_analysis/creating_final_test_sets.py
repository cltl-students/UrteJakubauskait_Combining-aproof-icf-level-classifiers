"""
This script processes a dataset of linguistic elements stored in a CSV file 
(for example, "negations.csv") and creates expanded and domain-specific test
sets.

For each row in the original CSV, the original row is kept (with auxiliary
columns removed). Additionally, a modified copy of the row is created using a
copy of the text column and a corresponding label column. The row ID is updated
to distinguish it from the original. Then, the expanded dataset is created by
combining the original and modified rows. A 'domain' column is extracted from
the source file names. Finally, separate CSV and pickle files are created for
each unique domain.
"""

import pandas as pd
import re
df = pd.read_csv("negations.csv")

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

expanded_df["domain"] = expanded_df["source_file"].apply(lambda x: re.search(r'_(\w{3})\.csv', x).group(1))

domains = expanded_df["domain"].unique()
for domain in domains:
    df_domain = expanded_df[expanded_df["domain"] == domain].reset_index(drop=True)

    df_domain.to_csv(f"{domain}_test_negations.csv", index=False)
    df_domain.to_pickle(f"{domain}_test_negations.pkl")
