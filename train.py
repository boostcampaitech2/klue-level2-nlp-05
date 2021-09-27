import os
import random
import re
import json
import glob
import multiprocessing

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import wandb

def set_all_seeds(seed, verbose = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    print("All random seeds set to", seed)

def parse_arguments(parser, verbose = False):

    # Set random seed
    parser.add_argument('--seed', type=int, default=None, help="random seed (default: None)")
    parser.add_argument('--verbose', type=str, default="n", choices=["y", "n"], help="verbose (default: n)")

    # Container environment
    parser.add_argument('--data_dir',  type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './saved'))
    parser.add_argument('--name', type=str, default="exp")
    parser.add_argument('--load_model', type=str, help="Load pretrained model if not None")

    # Load Dataset and construct DataLoader
    parser.add_argument('--dataset', type=str, default='BaseDataset', help="name of dataset (default: BaseDatset)")
    parser.add_argument('--batch_size', metavar='B', type=int, default=1, help="train set batch size (default: 1)")
    parser.add_argument('--val_ratio', type=float, default=0.0, help="valid set ratio (default: 0.0)")
    parser.add_argument('--val_batch_size', metavar='B', type=int, help="valid set batch size (default set to batch_size)")

    # Load model and set optimizer
    parser.add_argument('--model', type=str, default='BaseModel', help="model name (default: BaseModel)")
    parser.add_argument('--optim', type=str, default='SGD', help="optimizer name (default: SGD)")
    parser.add_argument('--momentum', type=float, default=0., help="SGD with momentum (default: 0.0)")

    # training setup
    parser.add_argument('--epochs', type=int, metavar='N', default=1, help="number of epochs (default 1)")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument('--log_every', type=int, metavar='N', default=1, help="log every N epochs")

    # Learning Rate Scheduler
    group_lr = parser.add_argument_group('lr_scheduler')
    group_lr.add_argument("--lr_type",  type=str, metavar='TYPE', default=None, help="lr scheduler type (default: None)")
    group_lr.add_argument("--lr_gamma", type=float, metavar='GAMMA', default=0.9, help="lr scheduler gamma (default: 0.9)")
    group_lr.add_argument("--lr_decay_step", type=int, metavar='STEP', default=10, help="lr scheduler decay step (default: 10)")

    # WanDB setup
    group_wandb = parser.add_argument_group('wandb')
    group_wandb.add_argument('--wandb_use', type=str, default="n", choices=["y", "n"], help="use wandb for logging (default: n)")
    group_wandb.add_argument('--wandb_project', type=str, metavar='PROJECT', default="exp", help="wandb project name")

    args = parser.parse_args()

    return args

def increment_path(path, overwrite=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        overwrite (bool): whether to overwrite or increment path (increment if False).
    """
    path = Path(path)
    
    if (path.exists() and overwrite) or (not path.exists()):
        if not os.path.exists(str(path).split('/')[0]):
            os.mkdir(str(path).split('/')[0])
        if not path.exists():
            os.mkdir(path)
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = f"{path}{n}"
        if not os.path.exists(path):
            os.mkdir(path)
        return path

def train(args, verbose=False):
    save_dir = increment_path(os.path.join(args.model_dir, args.name))
    if verbose: print("save_dir:", save_dir)

    use_cuda = torch.cuda.is_available() 
    device = torch.device('cuda:0' if use_cuda else 'cpu') 
    if verbose: print('training on:', device)

    # Dataset
    dataset_module = getattr(import_module("dataset.dataset"), args.dataset)
    # default: BaseDataset
    dataset = dataset_module(data_dir=args.data_dir,)

    # Augmentation
    # transform_module = None
    # if args.img_augmentation:
    #     transform_module = getattr(import_module("dataset.transform"), args.img_augmentation)
    #     transform = transform_module(
    #         resize=args.img_resize,
    #         mean=dataset.mean,
    #         std=dataset.std
    #     )
    #     dataset.set_transform(transform)

    # DataLoader
    # TODO: not spliting dataset (when args.val_ratio == 0)
    # train_ds, valid_ds = dataset.split_dataset(args.val_ratio)
    # pin_memory = use_cuda

    # train_dl = DataLoader(
    #     train_ds,
    #     batch_size=args.batch_size,
    #     num_workers=multiprocessing.cpu_count()//2,
    #     shuffle=True,
    #     pin_memory=pin_memory,
    #     drop_last=True,
    # )

    # valid_dl = DataLoader(
    #     valid_ds,
    #     batch_size=args.val_batch_size if args.val_batch_size else args.batch_size,
    #     num_workers=multiprocessing.cpu_count()//2,
    #     shuffle=False,
    #     pin_memory=pin_memory,
    #     drop_last=True,
    # )

    # Model
    # first looking for "model/{args.model}.py" and import {args.model} class
    # if not found, try "model/models.py" and import {args.model} class
    # try:
    #     model_module = getattr(import_module("model." + args.model), args.model)
    # except ModuleNotFoundError:
    #     model_module = getattr(import_module("model.models"), args.model)
    
    # model = model_module().to(device)

    # Load Saved Model
    # if {args.load_model} is given, then load the saved model from {args.model_dir}
    # if args.load_model is not None:
    #     saved_model = os.path.join(args.model_dir, args.load_model)
    #     model.load_state_dict(torch.load(saved_model, map_location=device))
    #     if verbose: print("succesfully loaded saved model from", saved_model)
    
    # model = torch.nn.DataParallel(model)

    # Loss & Metrics
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.SGD(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     momentum=args.momentum
    # )

    # LR Scheduler
    # scheduler = None
    # if args.lr_scheduler == "StepLR":
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, 
    #                                           args.lr_decay_step, 
    #                                           gamma=args.lr_gamma)
    # elif args.lr_scheduler == "ReduceLROnPlateau":
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # Logging
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
   
def main():
    parser = argparse.ArgumentParser(description="Train the model with the arguments given")
    args = parse_arguments(parser)
    v = args.verbose == "y"
    if v: print(args)

    if args.seed is not None:
        set_all_seeds(args.seed, verbose=v)

    if args.wandb_use == "y":
        wandb.init(project=args.wandb_project, 
                   config={"batch_size": args.batch_size,
                           "lr": args.lr,
                           "epochs": args.epochs,
                           "backbone": args.model,
                           "criterion_name": "CE",
                           "save_name": args.wandb_name})

    train(args, verbose=v)

if __name__ == '__main__':
    main()