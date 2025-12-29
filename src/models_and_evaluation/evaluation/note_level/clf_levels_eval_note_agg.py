"""
Aggregate sentence-level regression predictions to note-level and compute evaluation metrics.

The note-level labels (gold and predictions) are a mean of the sentence-level labels belonging to the same note.
The evaluation metrics include: mean absolute error, mean squared error, root mean squared error.

Evaluation metrics are computed and reported separately per domain.

$ clf_levels_eval_note_agg.py --doms ATT
"""

import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, '..')
from pathlib import Path

import pickle
import numpy as np

def evaluate(test_pkl, output_file, dom):
    """
    Process sentence-level predictions of a regression model to generate evaluation metrics on a note-level.
    The note-level labels (gold and predictions) are a mean of the sentence-level labels belonging to the
    same note. The evaluation metrics include: mean absolute error, mean squared error, root mean squared error.
    The values are printed to the command line.

    Parameters
    ----------
    test_pkl : str
        Path to a pickled pandas DataFrame containing sentence-level test data. The DataFrame must include a
        'labels' column with gold regression labels and a 'NotitieID' column identifying notes.
    output_file : str
        Path to a pickled object containing sentence-level model predictions, indexed by domain.
    dom : str
        Domain (ICF category) to evaluate.

    Returns
    -------
    None
    """

    # load data
    test = pd.read_pickle(test_pkl)

    # load sentence-level predictions
    with open(output_file, 'rb') as f:
        model_outputs = pickle.load(f)

    preds = model_outputs[dom]
    test['preds'] = preds

    labels = test.groupby('NotitieID').labels.mean()
    preds = test.groupby('NotitieID')['preds'].mean()
    df = pd.concat([labels, preds], axis = 1)
    print(f"{dom} Number of notes in the test set: {len(df)}")
    df = df.dropna()
    print(f"{dom} Number of notes with a gold label: {len(df)}")

    mse_val = mean_squared_error(df.labels, df.preds)
    rmse_val = np.sqrt(mse_val)

    print(f"{dom} mae: {round(mean_absolute_error(df.labels, df.preds), 2)}")
    print(f"{dom} mse: {round(mse_val, 2)}")
    print(f"{dom} rmse: {round(rmse_val, 2)}")

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data_expr_sept')
    argparser.add_argument('--doms', nargs='*', default=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'])
    argparser.add_argument('--testfile', default='test')
    argparser.add_argument('--outputfile', default='model_outputs')
    args = argparser.parse_args()
    
    print(f"Note-level metrics for {args.testfile}:")
    
    for dom in args.doms: 
        test_pkl = f"{args.testfile}.pkl"
        output_pkl = f"{args.outputfile}_{dom}.pkl"

        print(f"\n---{dom} ---")
        evaluate(test_pkl, output_pkl, dom)
