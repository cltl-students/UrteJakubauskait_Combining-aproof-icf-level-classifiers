"""
This script generates statistics for multiple training or development datasets
across  different domains, exports each domain’s dataset as a CSV, and writes a
summary report to a text file.

First of all, it iterates over a predefined list of domains (for example, 'CBP',
'FML', and 'HLC'). For each domain, it loads the corresponding dataset pickle
file (for example, 'dev_combined_ADM.pkl'). Then, it converts the dataset to a
DataFrame and exports it as a CSV file. Then, the script calculates basic
sentence length statistics: average and median sentence length (in words), and
shortest and longest sentence length. Then, it writes the statistics along with
column names and number of rows to a summary text file. Finally, it saves the
summary report to 'dev_combined_statistics.txt'.
"""

import pickle
import pandas as pd

domains = ["CBP", "FML", "HLC", "HRN", "HSP", "MAE", "SLP", "SOP", 
           "ADM", "ATT", "BER", "ENR", "ETN", "FAC", "INS", "MBW", "STM"]

output_file = "dev_combined_statistics.txt"

with open(output_file, "w", encoding="utf-8") as f_out:
    f_out.write("Statistics for development files\n")
    f_out.write("-" * 60 + "\n\n")

    for dom in domains:
        file_path = f"../data/data_expr_sept/dev_combined_{dom}.pkl"
        obj = pd.read_pickle(file_path)
        df = pd.DataFrame(obj)
        df.to_csv(f"../data/data_expr_sept/dev_combined_{dom}.csv", index=False)
        
        text_lengths = df["text"].astype(str).str.split().apply(len) 
        
        f_out.write(f"Domain: {dom}\n")
        f_out.write(f"File: {file_path}\n")
        f_out.write(f"Columns: {df.columns.tolist}\n")
        f_out.write(f"Number of rows: {len(df)}\n")
        f_out.write(f"Average sentence length (words): {text_lengths.mean()}\n")
        f_out.write(f"Median sentence length (words): {text_lengths.median()}\n")
        f_out.write(f"Shortest sentence (words): {len(df['text'].iloc[text_lengths.idxmin()])}\n")
        f_out.write(f"Longest sentence length (words): {len(df['text'].iloc[text_lengths.idxmax()])}\n")
        f_out.write("-" * 60 + "\n\n")

