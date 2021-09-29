import os
import random
import re
import json
import glob
import multiprocessing

import argparse
from importlib import import_module
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
import pickle as pickle

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BartModel, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TrainingArguments

from transformers.optimization import AdamW

from tqdm import tqdm
import wandb
import requests

######################################
### HELPER FUNCTIONS
######################################

def set_all_seeds(seed, verbose=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    if verbose:
        print("All random seeds set to", seed)


def parse_arguments(parser):

    # Set random seed
    parser.add_argument('--seed', type=int, default=None,
                        help="random seed (default: None)")
    parser.add_argument('--verbose', type=str, default="n",
                        choices=["y", "n"], help="verbose (default: n)")

    # Container environment
    parser.add_argument('--data_dir',  type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', './data'))
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', './saved'))
    parser.add_argument('--name', type=str, default="exp")
    parser.add_argument('--load_model', type=str,
                        help="Load pretrained model if not None")

    # Load Dataset and construct DataLoader
    parser.add_argument('--dataset', type=str, default='BaseDataset',
                        help="name of dataset (default: BaseDatset)")
    parser.add_argument('--batch_size', metavar='B', type=int,
                        default=1, help="train set batch size (default: 1)")
    parser.add_argument('--val_ratio', type=float, default=0.0,
                        help="valid set ratio (default: 0.0)")
    parser.add_argument('--val_batch_size', metavar='B', type=int,
                        help="valid set batch size (default set to batch_size)")

    # Load model and set optimizer
    parser.add_argument('--model', type=str, default='BaseModel',
                        help="model name (default: BaseModel)")
    parser.add_argument('--optim', type=str, default='SGD',
                        help="optimizer name (default: SGD)")
    parser.add_argument('--momentum', type=float, default=0.,
                        help="SGD with momentum (default: 0.0)")

    # training setup
    parser.add_argument('--epochs', type=int, metavar='N',
                        default=1, help="number of epochs (default 1)")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate (default: 1e-3)")
    parser.add_argument('--log_every', type=int, metavar='N',
                        default=1, help="log every N epochs")

    # Learning Rate Scheduler
    group_lr = parser.add_argument_group('lr_scheduler')
    group_lr.add_argument("--lr_type",  type=str, metavar='TYPE',
                          default=None, help="lr scheduler type (default: None)")
    group_lr.add_argument("--lr_gamma", type=float, metavar='GAMMA',
                          default=0.9, help="lr scheduler gamma (default: 0.9)")
    group_lr.add_argument("--lr_decay_step", type=int, metavar='STEP',
                          default=10, help="lr scheduler decay step (default: 10)")

    # WanDB setup
    group_wandb = parser.add_argument_group('wandb')
    group_wandb.add_argument('--wandb_use', type=str, default="n",
                             choices=["y", "n"], help="use wandb for logging (default: n)")
    group_wandb.add_argument('--wandb_project', type=str,
                             metavar='PROJECT', default="exp", help="wandb project name")

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


######################################
### KLUE SPECIFICS
######################################


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
                  'org:product', 'per:title', 'org:alternate_names',
                  'per:employee_of', 'org:place_of_headquarters', 'per:product',
                  'org:number_of_employees/members', 'per:children',
                  'per:place_of_residence', 'per:alternate_names',
                  'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
                  'per:spouse', 'org:founded', 'org:political/religious_affiliation',
                  'org:member_of', 'per:parents', 'org:dissolved',
                  'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
                  'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
                  'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }


def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


######################################
### DATA LOADER RELATED
######################################

# TODO: bucketed_batch_indicies 수정하기!
def bucketed_batch_indices(
    src_lens: pd.Series,
    tgt_lens: pd.Series,
    batch_size: int,
    max_pad_len: int
) -> List[List[int]]:
    assert len(src_lens) == len(tgt_lens)

    batch_map = defaultdict(list)
    batch_indices_list = []

    src_len_min = pd.Series.min(src_lens)
    tgt_len_min = pd.Series.min(tgt_lens)

    for idx, (src_len, tgt_len) in enumerate(zip(src_lens, tgt_lens)):
        src = (src_len - src_len_min + 1) // max_pad_len
        tgt = (tgt_len - tgt_len_min + 1) // max_pad_len
        batch_map[(src, tgt)].append(idx)

    for _, value in batch_map.items():
        batch_indices_list += [value[i:i+batch_size]
                               for i in range(0, len(value), batch_size)]

    random.shuffle(batch_indices_list)

    return batch_indices_list

# TODO: collate_fn 현 데이터셋에 맞춰 수정하기!
def collate_fn(
    batched_samples: List[Tuple[List[int], List[int], List[int]]],
    pad_token_idx
) -> Tuple[torch.Tensor, torch.Tensor]:

    PAD = pad_token_idx
    B = len(batched_samples)

    batched_samples = sorted(
        batched_samples, key=lambda x: x["src_idx"], reverse=True)

    src_sentences = []
    src_attention = []
    tgt_sentences = []

    for sample in batched_samples:
        src_sentences.append(torch.tensor(sample["src_idx"]))
        src_attention.append(torch.tensor(sample["src_attn"]))
        tgt_sentences.append(torch.tensor(sample["tgt_idx"]))

    src_sentences = torch.nn.utils.rnn.pad_sequence(
        src_sentences, padding_value=PAD, batch_first=True)
    src_attention = torch.nn.utils.rnn.pad_sequence(
        src_attention, padding_value=0, batch_first=True)
    tgt_sentences = torch.nn.utils.rnn.pad_sequence(
        tgt_sentences, padding_value=PAD, batch_first=True)

    assert src_sentences.size(0) == B and tgt_sentences.size(0) == B
    assert src_sentences.dtype == torch.long and tgt_sentences.dtype == torch.long
    return {'src_idx': src_sentences,
            'src_attn': src_attention,
            'tgt_idx': tgt_sentences}


def send_web_hooks(text, url):
    # Please keep your url privately
    payload = {"text": text}
    requests.post(url, json=payload)


def train(args, verbose=False):
    save_dir = increment_path(os.path.join(args.model_dir, args.name))
    if verbose:
        print("save_dir:", save_dir)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    if verbose:
        print('training on:', device)

    # Build Dataset
    dataset_module = None
    try:
        dataset_module = getattr(import_module(
            "dataset."+args.dataset), args.dataset)
    except:
        dataset_module = getattr(import_module(
            "dataset.dataset"), args.dataset)

    MAX_SEQ_LEN = 2048

    dataset = dataset_module(
        data_dir=args.data_dir,
        max_seq_len=MAX_SEQ_LEN,
        dropna=True)

    # Augmentation must be implemented in dataset!
    # not in train.py

    # DataLoader
    # TODO: train-valid split or building seperate valid data
    # TODO: not spliting dataset (when args.val_ratio == 0)
    # train_ds, valid_ds = dataset.split_dataset(args.val_ratio)

    batch_size = args.batch_size
    MAX_PAD_LEN = 10
    NUM_WORKERS = 2
    pin_memory = use_cuda

    # TODO: change it according to dataloader
    dataloader = DataLoader(dataset,
                            collate_fn=lambda x: collate_fn(x, dataset.tokenizer.pad_token_id), 
                            num_workers=NUM_WORKERS,
                            pin_memory=pin_memory,
                            batch_sampler=bucketed_batch_indices(dataset.data["len_text"],
                                                                 dataset.data["len_summary"],
                                                                 batch_size=batch_size,
                                                                 max_pad_len=MAX_PAD_LEN))

    # TODO: requires consensus on how to import model
    # Model

    # first looking for "model/{args.model}.py" and import {args.model} class
    # if not found, try "model/models.py" and import {args.model} class
    try:
        model_module = getattr(import_module(
            "model." + args.model), args.model)
    except ModuleNotFoundError:
        model_module = getattr(import_module("model.models"), args.model)

    model = model_module().to(device)

    # Load Saved Model
    # if {args.load_model} is given, then load the saved model from {args.model_dir}
    if args.load_model is not None:
        saved_model = os.path.join(args.model_dir, args.load_model)
        model.load_state_dict(torch.load(saved_model, map_location=device))
        if verbose:
            print("succesfully loaded saved model from", saved_model)

    # BART
    # model = None
    # if args.load_model:
    #     model = BartForConditionalGeneration.from_pretrained("./kobart_summary").to(device)
    # else:
    #     model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model()).to(device)

    # T5
    # model_name = 'KETI-AIR/ke-t5-base'
    # model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Optimizer
    # LayerNorm should not be decayed...
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = None
    if args.optim == "SGD":
        if args.momentum > 0.0:
            optimizer = optim.SGD(model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)

    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    elif args.optim == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.lr, correct_bias=False)

    # TODO: LR Scheduler
    scheduler = None
    if args.lr_scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              args.lr_decay_step,
                                              gamma=args.lr_gamma)
    elif args.lr_scheduler == "ReduceLROnPlateau":
        PATIENCE = 3
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         patience=PATIENCE)

    # Train
    num_epochs = args.epochs
    SAVE_EVERY = 1

    for epoch in range(num_epochs):

        if verbose:
            print("="*10, "epoch", epoch, "="*10)

        for sentences in tqdm(dataloader):

            # TODO: input change...
            src_ids = sentences['src_idx'].to(device)
            src_attn = sentences['src_attn'].to(device)
            tgt_ids = sentences['tgt_idx'].to(device)

            out = model(src_ids, src_attn, labels=tgt_ids)

            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()

        if ((epoch+1) % SAVE_EVERY == 0) or (epoch+1 == num_epochs):
            # torch.save(model, os.path.join(save_dir, args.name + str(epoch) + '.pkl'))
            model.save_pretrained(os.path.join(
                save_dir, "pretrained_" + args.name + str(epoch+1)))

    # Logging
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)


def main():
    v = True
    parser = argparse.ArgumentParser(
        description="Train the model with the arguments given")
    args = parse_arguments(parser, verbose=v)

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
