import argparse
import logging
import os
import json
import pathlib

from data.data_modules_synthetic import SyntheticSNPs
from src.run_experiment_synthetic import *
# from src.utils import *

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '-sc', '--scenario',
      type=str,
      help='Which max order of interaction to include: Linear or Squared',
      required=True,
    )

    parser.add_argument(
      '--source',
      type=str,
      help='Which type of synthetic data: random, numpy, Cordell',
      required=True,
    )

    parser.add_argument(
        '-m', '--model',
        default='torchLASSO',
        help='Selects model to use for subsequent training'
    )

    parser.add_argument(
        '--res',
        action='store_true',
        help='Predict residual of LASSO model'
    )

    parser.add_argument(
        '-S', '--seed',
        type=int,
        help='Random seed to use for the experiment',
        required=True
    )

    parser.add_argument(
        '-b', '--beta',
        type=float,
        default=0.1,
        help='Set the strength of the nonlinear interaction.'
    )

    parser.add_argument(
        '-nw', '--num_workers',
        type=int,
        default=3,
        help='The number of parallel processes to load the data. Default is 3.'
    )

    parser.add_argument(
        '-n_gpu', '--num_gpu',
        type=int,
        default=0,
        help='The number of gpus to use.'
    )

    parser.add_argument(
        '-dev_mode', '--dev_mode',
        action='store_true',
        help='The debug mode. If True then it only trains on 10 mini-batches'
    )

    parser.add_argument(
        '--save_model',
        action='store_true',
        help='Saving the model file for later ensemble methods.'
    )

    args = parser.parse_args()

    if args.model == 'Single_SNP_MLP':
        params = {'l':0.,#1e-5,  #SNP_MLP
                  'lr':1e-4,#1e-3,
                  'bs':256,#128,
                  'p':0.0,
                  'dropout_all':True,
                  'activation':'ReLU',#'Softplus',
                  'n_layers':3,
                  'layer_0':50,
                  'layer_1':200,
                  'layer_2':200,
                  'scenario':args.scenario,
                  'beta':args.beta}
    elif args.model == 'torchLASSO':
        params = {'l':0.,#'l':1e-3,     #Linear Model
                  'lr':1e-4,
                  'bs':256,
                  'scenario':args.scenario,
                  'beta':args.beta}

    # params = {'l':1e-3,  #WideSNPResNet
    #           'lr':1e-4,
    #           'bs':256,
    #           'p':0.5,
    #           'dropout_all':True,
    #           'activation':'ReLU',
    #           'n_res_layers':3}
    results = run_experiment(params,args)

    print(f'Results:{results}')
