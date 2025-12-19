"""
Aggregate sentence-level regression predictions to note-level and compute evaluation metrics.

The note-level labels (gold and predictions) are a mean of the sentence-level labels belonging to the same note.
The evaluation metrics include: mean absolute error, mean squared error, root mean squared error.

Evaluation metrics are computed and reported separately per domain.

$ python evaluate_model.py --doms ATT
"""

import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, '..')
from pathlib import Path

import pickle
import numpy as np

def evaluate(test_pkl, output_pkl, domains):
    """
    Process sentence-level predictions of a regression model to generate evaluation metrics on a note-level.
    The note-level labels (gold and predictions) are a mean of the sentence-level labels belonging to the same note.
    The evaluation metrics include: mean absolute error, mean squared error, root mean squared error. The values are printed to the command line.

    Parameters
    ----------
    test_pkl: str
        path to pickled df with the training data, which must contain a 'labels' column and a column with predictions (whose name is given by the `pred_col` argument); both columns contain numeric values on a continuous scale
    pred_col: str
        the name of the column containing the predictions; its format is "preds_{name_of_the_model}"

    Returns
    -------
    None
    """

    # load data
    test = pd.read_pickle(test_pkl)

    # load sentence-level predictions
    with open(output_pkl, 'rb') as f:
         model_outputs = pickle.load(f)

    preds = model_outputs[dom]
    test['preds'] = preds 

    labels = test.groupby('NotitieID').labels.mean()
    preds = test.groupby('NotitieID')['preds'].mean()
    df = pd.concat([labels, preds], axis = 1)

    print(f"{dom} Number of notes in the test set: {len(df)}")
    print(df)
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

    for dom in args.doms:
        test_pkl = f"../data_expr_sept/clf_levels_{dom}_sents/{args.testfile}.pkl"
        output_pkl = f"../models/levels_all_tokens_sents_combined_all/eval_test_all_tokens_combined_all/{args.outputfile}_{dom}.pkl"
    
    print(f"Note-level metrics for {args.testfile}:")
    evaluate(test_pkl, output_pkl, args.doms)
