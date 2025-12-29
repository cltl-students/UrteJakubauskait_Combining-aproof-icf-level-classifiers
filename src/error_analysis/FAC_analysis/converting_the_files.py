"""
This script converts a Python pickle file (.pkl) into a CSV file (.csv)
using pandas.
"""

import pickle
import pandas as pd 

pkl_file = "YOUR_INPUT_FILE.pkl"
csv_file = "YOUR_OUTPUT_FILE.csv"

df = pd.read_pickle(pkl_file)
df.to_csv(csv_file, index=False)

with open(pkl_file, "rb") as f:
    data = pickle.load(f)

if isinstance(data, pd.DataFrame):
    data.to_csv(csv_file, index=False)
else:
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
