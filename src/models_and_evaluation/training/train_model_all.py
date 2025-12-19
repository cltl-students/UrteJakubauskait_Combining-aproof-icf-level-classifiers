"""
Fine-tune and save a single regression model predicting functioning levels 
across all domains jointly using Simple Transformers.

The script can be customized with the following parameters:
    --datapath: data dir
    --config: json file containing the model args
    --model_type: type of the pre-trained model, e.g. bert, roberta, electra
    --modelpath: models dir
    --model_name: the pre-trained model, either from Hugging Face or locally stored
    --hf: pass this parameter if a model from Hugging Face is used
    --clas_unit: classification unit ('sent' or 'note')
    --train_on: name of the file with the train data
    --eval_on: name of the file with the eval data

To change the default values of a parameter, pass it in the command line, e.g.:

$ python train_model_all.py --clas_unit note
"""

import argparse
import logging
import warnings
import json
import torch
import pandas as pd
from simpletransformers.classification import ClassificationModel

import sys
from pathlib import Path
sys.path.insert(0, '..')

def train(train_pkl,
          eval_pkl,
          config_json,
          args_key,
          model_type,
          model_name):
    
    """
    Fine-tune and save a regression model with Simple Transformers.

    Parameters
    ----------
    train_pkl: str
        path to pickled df with the training data, which must contain the columns 'text' and 'labels'; the labels are numeric values on a continuous scale
    eval_pkl: {None, str}
        path to pickled df for evaluation during training (optional)
    config_json: str
        path to a json file containing the model args
    args: str
        the name of the model args dict from `config_json` to use
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: str
        the exact architecture and trained weights to use; this can be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model file

    Returns
    -------
    None
    """
    
    # check CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        def custom_formatwarning(msg, *args, **kwargs):
            return str(msg) + '\n'
        warnings.formatwarning = custom_formatwarning
        warnings.warn('CUDA device not available; running on a CPU!')

    # logging
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.WARNING)

    # load data
    train_data = pd.read_pickle(train_pkl)
    eval_data = pd.read_pickle(eval_pkl)

    # model args
    with open(str(config_json), 'r') as f:
        config = json.load(f)
    model_args = config[args_key]

    # model
    model = ClassificationModel(model_type,
                                model_name,
                                num_labels = 1,
                                args=model_args,
                                use_cuda=cuda_available)

    #train
    if model.args.evaluate_during_training:
        model.train_model(train_data, eval_df=eval_data)
    else:
        model.train_model(train_data)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data_expr_sept', help='must be listed as a key in /config.ini')
    argparser.add_argument('--doms', nargs='*', default=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'])
    argparser.add_argument('--config', default='config.json')
    argparser.add_argument('--model_type', default='roberta')
    argparser.add_argument('--modelpath', default='models')
    argparser.add_argument('--model_name', default='clin_nl_from_scratch')
    argparser.add_argument('--hf', dest='hugging_face', action='store_true')
    argparser.set_defaults(hugging_face=False)
    argparser.add_argument('--clas_unit', default='sent')
    argparser.add_argument('--train_on', default='train')
    argparser.add_argument('--eval_on', default='dev')
    args = argparser.parse_args()

    train_dfs = []
    eval_dfs = []

    CURRENT_DIR = Path(__file__).parent
    PROJECT_ROOT = CURRENT_DIR.parent
    BASE_DATA_PATH = PROJECT_ROOT / args.datapath

    for dom in args.doms:
        domain_dir = BASE_DATA_PATH / f"clf_levels_{dom}_{args.clas_unit}s"
        train_pkl_dom = domain_dir / f"{args.train_on}.pkl"
        eval_pkl_dom = domain_dir / f"{args.eval_on}.pkl"

        train_df_dom = pd.read_pickle(train_pkl_dom)
        eval_df_dom = pd.read_pickle(eval_pkl_dom)

        #converting to a DataFrame (only for combined files as they were saved differently)
        train_df_dom = pd.DataFrame(train_df_dom)
        eval_df_dom = pd.DataFrame(eval_df_dom)

        if 'domain' not in train_df_dom.columns:
            train_df_dom['domain'] = dom
            eval_df_dom['domain'] = dom

        train_dfs.append(train_df_dom)
        eval_dfs.append(eval_df_dom)

    # combine all files into one and shuffle the rows
    train_df_all = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    eval_df_all = pd.concat(eval_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    combined_dir = BASE_DATA_PATH / f"clf_levels_all_{args.clas_unit}s"
    combined_dir.mkdir(parents=True, exist_ok=True)
    train_pkl = combined_dir / f"{args.train_on}.pkl"
    eval_pkl = combined_dir / f"{args.eval_on}.pkl"

    train_df_all.to_pickle(train_pkl)
    eval_df_all.to_pickle(eval_pkl)

    model_args = f"levels_all_{args.clas_unit}s"

    # model stored locally (default) or on HuggingFace (--hf)
    model_name = str((PROJECT_ROOT / args.modelpath / args.model_name))
    if args.hugging_face:
        model_name = args.model_name

    config_path = CURRENT_DIR / args.config

    print(f"TRAINING {model_args}")
    train(train_pkl,
          eval_pkl,
          config_path,
          model_args,
          args.model_type,
          model_name)
