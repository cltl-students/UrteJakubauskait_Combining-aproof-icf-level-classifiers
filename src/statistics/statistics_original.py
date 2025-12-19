"""
This script performs an exploratory overview of a training, development, or test
dataset stored in a pickle file and exports it as a CSV file.

It loads the dataset from a pickle file (for example, 'train.pkl'). Then,
it exports the dataset to a CSV file (for example, 'train_ADM.csv'). Finally, it
prints basic dataset statistics, including: column names, number of rows,
distribution of annotators, distribution of labels, average and median sentence
length ('len_text'), and minimum and maximum sentence length.
"""

import pickle
import pandas as pd

df = pd.read_pickle("/A-Proof/data2/a-proof-zonmw/data_expr_sept/clf_levels_ADM_sents/train.pkl")
df.to_csv("train_ADM.csv", index=False)

print('Column names:')
print(df.columns.tolist())

print("\nNumber of Rows:")
print(len(df))

print("\nAnnotator Distribution:")
print(df['annotator'].value_counts())

print("\nLabels:")
print(df['labels'].value_counts())

print("\nAverage length:")
print(df['len_text'].mean())

print("\nMean length:")
print(df['len_text'].median())

print("\nSmallest and highest value:")
print(df['len_text'].min())
print(df['len_text'].max())
