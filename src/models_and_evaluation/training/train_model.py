"""
Fine-tune and save regression models predicting functioning levels per domain with Simple Transformers.

The script can be customized with the following parameters:
    --datapath: data dir (relative to project root, e.g. data_expr_sept)
    --doms: the domains for which models are trained
    --config: json file containing the model args
    --model_type: type of the pre-trained model, e.g. bert, roberta, electra
    --modelpath: models dir
    --model_name: the pre-trained model, either from Hugging Face or locally stored
    --hf: pass this parameter if a model from Hugging Face is used
    --clas_unit: classification unit ('sent' or 'note')
    --train_on: name of the file with the train data
    --eval_on: name of the file with the eval data

To change the default values of a parameter, pass it in the command line, e.g.:

$ python train_model.py --doms ATT INS FAC --clas_unit note
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
          args,
          model_type,
          model_name):
    
    """
    Fine-tune and save a regression model with Simple Transformers.

    Parameters
    ----------
    train_pkl: str or Path
        path to pickled df with the training data, which must contain the columns 'text' and 'labels'; the labels are numeric values on a continuous scale
    eval_pkl: {None, str, Path}
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

    # converting to a DataFrame (only for combined files as they were saved differently)
    train_data = pd.DataFrame(train_data)
    eval_data = pd.DataFrame(eval_data)

    # model args
    with open(str(config_json), 'r') as f:
        config = json.load(f)
    model_args = config[args]

    # model
    model = ClassificationModel(model_type,
                                model_name,
                                num_labels=1,
                                args=model_args,
                                use_cuda=cuda_available)

    # train
    if model.args.evaluate_during_training:
        model.train_model(train_data, eval_df=eval_data, show_running_loss=True)
    else:
        model.train_model(train_data, show_running_loss=True)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data_expr_sept', 
                           help='data folder name relative to project root, e.g. data_expr_sept or data_expr_july')
    argparser.add_argument('--doms', nargs='*', default=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'])
    argparser.add_argument('--config', default='config.json')
    argparser.add_argument('--model_type', default='roberta')
    argparser.add_argument('--modelpath', default='models')
    argparser.add_argument('--model_name', default='clin_nl_from_scratch')
    argparser.add_argument('--hf', dest='hugging_face', action='store_true')
    argparser.set_defaults(hugging_face=False)
    argparser.add_argument('--clas_unit', default='sent')
    argparser.add_argument('--train_on', default='train')
    argparser.add_argument('--eval_on', default='dev', help='only used if `evaluate_during_training` is True in the model args in `config`')
    args = argparser.parse_args()

    # Resolve paths relative to the project root
    CURRENT_DIR = Path(__file__).parent   # directory of this script
    PROJECT_ROOT = CURRENT_DIR.parent     # adjust if the script is nested differently
    BASE_DATA_PATH = PROJECT_ROOT / args.datapath

    for dom in args.doms:

        domain_dir = BASE_DATA_PATH / f"clf_levels_{dom}_{args.clas_unit}s"
        train_pkl = domain_dir / f"{args.train_on}.pkl"
        eval_pkl = domain_dir / f"{args.eval_on}.pkl"

        model_args = f"levels_{dom.lower()}_{args.clas_unit}s"

        # model stored locally (default) or on HuggingFace (--hf)
        model_name = str((PROJECT_ROOT / args.modelpath) / args.model_name)
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
