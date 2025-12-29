"""
Evaluate fine-tuned regression model on an evaluation set for a single,
hardcoded domain.

Note: By default, the domain is hardcoded in the script (for example,
'dom = "STM"'). If you want to evaluate another domain, you need to modify the
'dom' variable in the script accordingly. This script does not loop over
multiple domains automatically.

Save the following outputs:
- evaluation metrics: MSE, RMSE, MAE, eval_loss
- model outputs
- wrong predictions

The script can be customized with the following parameters:
    --datapath: data dir
    --model_type: type of the fine-tuned model, e.g. bert, roberta, electra
    --modelpath: models dir
    --clas_unit: classification unit ('sent' or 'note')
    --eval_on: name of the file with the eval data

Example:
$ python evaluate_model_all.py --clas_unit note
"""

import argparse
import pickle
import warnings
import torch
import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, '..')
from pathlib import Path

def evaluate(test_pkl,
             model_type,
             model_name,
             output_dir):
    
    """
    Evaluate a fine-tuned regression model on a test set.
    Save evaluation metrics, model outputs and wrong predictions in 'output_dir'. The evaluation metrics include: MSE, RMSE, MAE and eval_loss.

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
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # load data
    test_data = pd.read_pickle(test_pkl)

    # check CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        def custom_formatwarning(msg, *args, **kwargs):
            return str(msg) + '\n'
        warnings.formatwarning = custom_formatwarning
        warnings.warn('CUDA device not available; running on CPU!')

    # load model
    model = ClassificationModel(model_type,
                                model_name,
                                num_labels=1,
                                use_cuda=cuda_available)

    domain_results = {}

    # evaluate model
    def calculate_rmse(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    dom = 'STM'
    dom_df = test_data[test_data['domain'] == dom]
    print(f"Evaluating domain {dom} with {len(dom_df)} rows...")

    result, model_outputs, wrong_preds = model.eval_model(dom_df)

    true_labels = np.array(dom_df['labels'].tolist())
    preds = np.array(model_outputs)

    mse = mean_squared_error(true_labels, preds)
    mae = mean_absolute_error(true_labels, preds)
    rmse = calculate_rmse(true_labels, preds)

    domain_results[dom] = {'MSE': mse,
                           'MAE': mae,
                           'RMSE': rmse,
                           'model_outputs': model_outputs,
                           'wrong_predictions': wrong_preds}
    
    print(f"{dom} -> MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
    # save evaluation outputs
    metrics_txt = output_dir / f"metrics_{dom}.txt"
    all_metrics = []

    print(f"Saving metrics...")

    with open(metrics_txt, 'w') as f:
        for dom, metrics in domain_results.items():
            f.write(f"Domain: {dom}\n")
            f.write(f" MSE: {metrics['MSE']:.4f}\n")
            f.write(f" MAE: {metrics['MAE']:.4f}\n")
            f.write(f" RMSE: {metrics['RMSE']:.4f}\n\n")

    with open(output_dir / f'model_outputs_{dom}.pkl', 'wb') as f:
        pickle.dump({dom: d['model_outputs'] for dom, d in domain_results.items()}, f)

    with open(output_dir / f'wrong_preds_{dom}.pkl', 'wb') as f:
        pickle.dump({dom: d['wrong_predictions'] for dom, d in domain_results.items()}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default='data_expr_sept', help='Must be listed as a key in config.ini')
    parser.add_argument('--doms', nargs='*', default=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'])
    parser.add_argument('--model_type', default='roberta')
    parser.add_argument('--modelpath', default='models')
    parser.add_argument('--model_name', default='levels_all_sents')
    parser.add_argument('--clas_unit', default='sent')
    parser.add_argument('--eval_on', default='test')
    args = parser.parse_args()

    CURRENT_DIR = Path(__file__).parent
    PROJECT_ROOT = CURRENT_DIR.parent
    BASE_DATA_PATH = PROJECT_ROOT / args.datapath

    test_dfs = []

    for dom in args.doms:
        domain_dir = BASE_DATA_PATH / f"clf_levels_{dom}_{args.clas_unit}s"
        test_pkl_dom = domain_dir / f"{args.eval_on}.pkl"

        df_dom = pd.read_pickle(test_pkl_dom)

        if 'domain' not in df_dom.columns:
            df_dom['domain'] = dom

        test_dfs.append(df_dom)

    # combine
    combined_test_df = pd.concat(test_dfs, ignore_index=True).reset_index(drop=True)

    combined_dir = BASE_DATA_PATH / f"clf_levels_all_{args.clas_unit}s"
    combined_test_path = combined_dir / f"{args.eval_on}.pkl"
    combined_test_df.to_pickle(combined_test_path)

    test_pkl = combined_test_path

    model_name = PROJECT_ROOT / args.modelpath / f"levels_all_sents_combined_all"
    output_dir = model_name / f'eval_{args.eval_on}_combined_all'

    print(f"Evaluating {model_name} on {test_pkl}")
    evaluate(test_pkl,
             args.model_type,
             str(model_name),
             output_dir)
