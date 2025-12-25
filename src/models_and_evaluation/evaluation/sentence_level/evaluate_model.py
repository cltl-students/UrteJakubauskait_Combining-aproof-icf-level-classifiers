"""
Evaluate fine-tuned regression models on an evaluation set.

Save the following outputs per model:
- evaluation metrics: MSE, RMSE, MAE, eval_loss
- model outputs
- wrong predictions

The script can be customized with the following parameters:
    --datapath: data dir
    --doms: the domains for which models are evaluated
    --model_type: type of the fine-tuned model, e.g. bert, roberta, electra
    --modelpath: models dir
    --clas_unit: classification unit ('sent' or 'note')
    --eval_on: name of the file with the eval data

To change the default values of a parameter, pass it in the command line, e.g.:

$ python evaluate_model.py --doms ATT INS FAC --clas_unit note
"""

import argparse
import pickle
import warnings
import torch
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, '..')
from pathlib import Path

import numpy as np

def evaluate(test_pkl,
             model_type,
             model_name,
             output_dir):
    
    """
    Evaluate a fine-tuned regression model on a test set.
    Save evaluation metrics, model outputs and wrong predictions in `output_dir`. The evaluation metrics include: MSE, RMSE, MAE and eval_loss.

    Parameters
    ----------
    test_pkl: str
        path to pickled df with the test data, which must contain the columns 'text' and 'labels'; the labels are numeric values on a continuous scale
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: str
        path to a directory containing model file
    output_dir: Path
        path to a directory where outputs should be saved

    Returns
    -------
    None
    """

    # load data
    test_data = pd.read_pickle(test_pkl)

    # check CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        def custom_formatwarning(msg, *args, **kwargs):
            return str(msg) + '\n'
        warnings.formatwarning = custom_formatwarning
        warnings.warn('CUDA device not available; running on a CPU!')
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # load model
    model = ClassificationModel(model_type,
                                model_name,
                                num_labels=1,
                                use_cuda=cuda_available)

    # evaluate model
    def calculate_rmse(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    result, model_outputs, wrong_predictions = model.eval_model(
        test_data,
        output_dir=str(output_dir),
        mae=mean_absolute_error,
        mse=mean_squared_error,
        rmse=calculate_rmse)

    # save evaluation outputs
    model_outputs_path = output_dir / 'model_outputs.pkl'
    with open(model_outputs_path, 'wb') as f:
        pickle.dump(model_outputs, f)

    wrong_predictions_path = output_dir / 'wrong_preds.pkl'
    with open(wrong_predictions_path, 'wb') as f:
        pickle.dump(wrong_predictions, f)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data_expr_sept', help='must be listed as a key in /config.ini')
    argparser.add_argument('--doms', nargs='*', default=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'])
    argparser.add_argument('--model_type', default='roberta')
    argparser.add_argument('--modelpath', default='models')
    argparser.add_argument('--clas_unit', default='sent')
    argparser.add_argument('--eval_on', default='test')
    args = argparser.parse_args()

    CURRENT_DIR = Path(__file__).parent
    PROJECT_ROOT = CURRENT_DIR.parent
    BASE_DATA_PATH = PROJECT_ROOT / args.datapath

    for dom in args.doms:
        domain_dir = BASE_DATA_PATH / f"clf_levels_{dom}_{args.clas_unit}s"
        test_pkl = domain_dir / f"{args.eval_on}.pkl"

        model_name = PROJECT_ROOT / args.modelpath / f"levels_{dom.lower()}_{args.clas_unit}s_combined"
        test_output_dir = model_name / f'eval_{args.eval_on}_{dom}_combined'

        print(f"Evaluating {model_name} on {args.eval_on}.pkl")
        evaluate(test_pkl,
                 args.model_type,
                 str(model_name),
                 test_output_dir)
