import os
import random
import re
import json
import glob
import multiprocessing

import argparse
from importlib import import_module
from pathlib import Path
from typing import Union, List, Tuple
from collections import defaultdict
import pickle as pickle

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from transformers import T5Tokenizer, T5ForConditionalGeneration

from transformers.optimization import AdamW

from tqdm import tqdm
from transformers.utils.dummy_pt_objects import ModalEmbeddings
import wandb
import requests

######################################
# HELPER FUNCTIONS
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
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/dataset'))
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', './saved'))
    parser.add_argument('--log_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', './logs'))
    parser.add_argument('--name', type=str, default="exp",
                        help='name of the custom model and experiment')
    parser.add_argument('--load_model', type=str,
                        help="Load pretrained model if not None")

    # Load Dataset and construct DataLoader
    parser.add_argument('--dataset', type=str, default='BaselineDataset',
                        help="name of dataset (default: BaselineDataset)")
    parser.add_argument('--additional', type=str, nargs='*',
                        help="list of additional dataset file names")
    parser.add_argument('--batch_size', metavar='B', type=int,
                        default=1, help="train set batch size (default: 1)")
    parser.add_argument('--val_file', type=str, choices=["y", "n"],
                        default="n", help="whether to use valid.csv file (default: n)")
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help="valid set ratio (default: 0.2)")
    parser.add_argument('--val_batch_size', metavar='B', type=int,
                        help="valid set batch size (default set to batch_size)")

    # Preprocessor and Data Augmentation
    parser.add_argument('--preprocessor', type=str, default='BaselinePreprocessor',
                        help="type of preprocessor (default: BaselinePreprocessor)")
    parser.add_argument('--augmentation', type=str,
                        help="type of augmentation (default: None)")

    # Load model and set optimizer
    parser.add_argument('--model', type=str, default='BaseModel',
                        help="model name (default: BaseModel)")
    parser.add_argument('--num_labels', type=int, default=30,
                        help="number of labels for classification (default: 30)")
    parser.add_argument('--optim', type=str, default='AdamW',
                        help="optimizer name (default: AdamW)")
    parser.add_argument('--momentum', type=float, default=0.,
                        help="SGD with momentum (default: 0.0)")

    # training setup
    parser.add_argument('--epochs', type=int, metavar='N',
                        default=1, help="number of epochs (default 1)")
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="learning rate (default: 1e-5)")
    parser.add_argument('--max_seq_len', type=int, metavar='L',
                        default=256, help="max sequence length (default 256)")
    parser.add_argument('--max_pad_len', type=int, metavar='L',
                        default=8, help="max padding length for bucketing (default 8)")
    parser.add_argument('--log_every', type=int, metavar='N',
                        default=500, help="log every N steps (default: 500)")
    parser.add_argument('--eval_every', type=int, metavar='N',
                        default=500, help="evaluation interval for every N steps (default: 500)")
    parser.add_argument('--save_every', type=int, metavar='N',
                        default=500, help="save model interval for every N steps (default: 500)")

    # Learning Rate Scheduler
    group_lr = parser.add_argument_group('lr_scheduler')
    group_lr.add_argument("--lr_type",  type=str, metavar='TYPE',
                          default="constant", help="lr scheduler type (default: constant)")
    group_lr.add_argument("--lr_weight_decay", type=float, metavar='LAMBDA',
                          default=0.01, help="weight decay rate for AdamW (default: 0.01)")
    group_lr.add_argument("--lr_gamma", type=float, metavar='GAMMA',
                          default=0.95, help="lr scheduler gamma (default: 0.95)")
    group_lr.add_argument("--lr_decay_step", type=int, metavar='STEP',
                          default=100, help="lr scheduler decay step (default: 100)")
    group_lr.add_argument("--lr_warmups", type=int, metavar='N',
                          default=500, help="lr scheduler warmup steps (default: 500)")

    args = parser.parse_args()

    return args


def increment_path(path, overwrite=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        overwrite (bool): whether to overwrite or increment path (increment if False).

    Returns:
        path: new path
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
# KLUE SPECIFICS
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
    return metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = metrics.precision_recall_curve(
            targets_c, preds_c)
        score[c] = metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = metrics.accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

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
# DATA LOADER RELATED
######################################

# TODO: bucketed_batch_indicies 수정하기!
def bucketed_batch_indices(
    src_lens: Union[List[int], np.ndarray, pd.Series],
    batch_size: int,
    max_pad_len: int
) -> List[List[int]]:

    batch_map = defaultdict(list)
    batch_indices_list = []

    src_len_min = np.min(src_lens)

    for idx, src_len in enumerate(src_lens):
        src = (src_len - src_len_min + 1) // max_pad_len
        batch_map[src].append(idx)

    for _, value in batch_map.items():
        batch_indices_list += [value[i:i+batch_size]
                               for i in range(0, len(value), batch_size)]

    random.shuffle(batch_indices_list)

    return batch_indices_list

# TODO: collate_fn 현 데이터셋에 맞춰 수정하기!
# we don't need collate_fn
# since huggingface automatically creates default collate function
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


def get_model_and_tokenizer(args, **kwargs):
    # Here, you also need to define tokenizer as well
    # since the type of tokenizer depends on the model

    NUM_LABELS = 30
    model = None
    tokenizer = None

    if args.model.lower().count("klue/bert-base"):
        MODEL_NAME = "klue/bert-base"
        LOAD_MODEL = args.load_model if args.load_model else MODEL_NAME

        if args.load_model:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.load_model)
            try:
                tokenizer = AutoTokenizer.from_pretrained(LOAD_MODEL)
            except:
                # in case, pretrained tokenizer doesn't exists
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        else:
            model_config = AutoConfig.from_pretrained(MODEL_NAME)
            model_config.num_labels = NUM_LABELS
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, config=model_config)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    elif args.model.lower().count("ke-t5"):
        MODEL_NAME = ""
        CLASS_NAME = "T5EncoderForSequenceClassificationMeanSubmeanObjmean"

        if args.model.count("large"):
            MODEL_NAME = 'KETI-AIR/ke-t5-large'
        elif args.model.count("small"):
            MODEL_NAME = 'KETI-AIR/ke-t5-small'
        else:
            MODEL_NAME = 'KETI-AIR/ke-t5-base'
        
        if args.load_model:
            LOAD_MODEL = args.load_model
            config = AutoConfig.from_pretrained(LOAD_MODEL)
        else:
            LOAD_MODEL = MODEL_NAME
            config = AutoConfig.from_pretrained(LOAD_MODEL)
            config.num_labels = 30
            config.dropout_p = 0.4
            config.focal_loss = False

        model_module = getattr(import_module("model.models"), CLASS_NAME)
        model = model_module(config)

        try:
            tokenizer = T5Tokenizer.from_pretrained(LOAD_MODEL)
        except:
            tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    elif args.model.lower().count("klue/roberta"):
        MODEL_NAME = ""

        if args.model.count("large"):
            MODEL_NAME = "klue/roberta-large"
        elif args.model.count("small"):
            MODEL_NAME = "klue/roberta-small"
        else:
            MODEL_NAME = "klue/roberta-base"
        
        LOAD_MODEL = args.load_model if args.load_model else MODEL_NAME

        if args.load_model:
            model = AutoModelForSequenceClassification.from_pretrained(LOAD_MODEL)
            try:
                tokenizer = AutoTokenizer.from_pretrained(LOAD_MODEL)
            except:
                # in case, pretrained tokenizer doesn't exists
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                    model_input_names = ["input_ids", "attention_mask"])
        else:
            model_config = AutoConfig.from_pretrained(MODEL_NAME)
            model_config.num_labels = 30
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, config=model_config)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                model_input_names = ["input_ids", "attention_mask"])

    else:
        # If the model is not specified above,
        # it first tries to look up for "model/{args.model}.py" and "model/models.py" file.
        # Additional setting should be provided with kwargs above.

        # If still not found, it tries to find the model in huggingface
        # with AutoModelForSequenceClassification & AutoTokenizer

        try:
            model_module = getattr(import_module(
                "model."+args.model), args.model)
            model = model_module()
            tokenizer = model.tokenizer

        except:

            try:
                model_module = getattr(
                    import_module("model.models"), args.model)
                model = model_module()
                tokenizer = model.tokenizer

            except:
                MODEL_NAME = args.model

                model_config = AutoConfig.from_pretrained(MODEL_NAME)
                model_config.num_labels = 30

                model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME, config=model_config)
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer


def train(args, verbose: bool=True):
    # Create folder
    SAVE_DIR = increment_path(os.path.join(args.model_dir, args.name))
    LOG_DIR = increment_path(os.path.join(args.log_dir, args.name))
    if verbose:
        print("save_dir:", SAVE_DIR)
        print("log_dir: ", LOG_DIR)

    # Device setting
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    if verbose:
        print('training on:', device)

    # Load Model & Tokenizer
    # because the type of tokenizer depends on the model
    model, tokenizer = get_model_and_tokenizer(args)
    model.to(device)

    # Build Dataset
    try:
        dataset_module = getattr(import_module(
            "dataset."+args.dataset), args.dataset)
    except:
        dataset_module = getattr(import_module(
            "dataset.dataset"), args.dataset)

    MAX_SEQ_LEN = args.max_seq_len
    NUM_LABELS = args.num_labels
    # max_length sometimes refers to maximum length in text generation
    # so, I used MAX_SEQ_LEN to indicate maximum input length fed to the model

    dataset, train_dataset, valid_dataset = None, None, None

    if args.val_file == "y":
        train_dataset = dataset_module(
            data_dir=args.data_dir,
            max_length=MAX_SEQ_LEN,
            num_labels=NUM_LABELS,
            additional=args.additional,
            valid=False,
            dropna=True)

        valid_dataset = dataset_module(
            data_dir=args.data_dir,
            max_length=MAX_SEQ_LEN,
            num_labels=NUM_LABELS,
            additional=args.additional,
            valid=True,
            dropna=True)

        if verbose:
            print("="*20)
            print("train-valid split to train:", len(train_dataset), "valid:", len(valid_dataset))
            print("train:")
            print(train_dataset.data['label'].value_counts())
            print("test:")
            print(valid_dataset.data['label'].value_counts())
            print("="*20)

    else:
        dataset = dataset_module(
            data_dir=args.data_dir,
            max_length=MAX_SEQ_LEN,
            num_labels=NUM_LABELS,
            additional=args.additional,
            dropna=True)
    # dataset must return
    # dict containing at least {'input_ids', 'attention_mask', 'labels'}
    # in order to work properly

    # TODO: Build Preprocessor
    preprocessor = None
    if args.preprocessor:
        try:
            preprocessor_module = getattr(import_module(
                "dataset.preprocessor."+args.preprocessor), args.preprocessor)
        except:
            preprocessor_module = getattr(import_module(
                "dataset.preprocessor.preprocessors"), args.preprocessor)

        preprocessor = preprocessor_module()

    # Build Augmentation
    # unk, RE, RI, ...
    # this result will be fixed for entire training steps
    augmentation = None
    if args.augmentation:
        try:
            augmentation_module = getattr(import_module(
                "dataset.augmentation."+args.augmentation), args.augmentation)
        except:
            augmentation_module = getattr(import_module(
                "dataset.augmentation.augmentations"), args.augmentation)

        augmentation = augmentation_module(tokenizer)

    
    if dataset is not None:
        dataset.set_tokenizer(tokenizer)
        dataset.set_preprocessor(preprocessor)
        if augmentation is not None:
            dataset.set_augmentation(augmentation)
        dataset.preprocess()

    if train_dataset is not None:
        train_dataset.set_tokenizer(tokenizer)
        train_dataset.set_preprocessor(preprocessor)
        if augmentation is not None:
            train_dataset.set_augmentation(augmentation)
        train_dataset.preprocess()
            

    if valid_dataset is not None:
        valid_dataset.set_tokenizer(tokenizer)
        valid_dataset.set_preprocessor(preprocessor)
        if augmentation is not None:
            valid_dataset.set_augmentation(augmentation)
        valid_dataset.preprocess()

    
    added_token_num = train_dataset.get_special_token_num()
    if added_token_num > 0:
        model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)

    # TODO: train-valid split
    # TODO: do not split (= train with whole data) if val_ratio == 0.0

    if args.val_ratio > 0.0 and dataset is not None:
        train_ids, valid_ids = train_test_split(list(range(len(dataset.data))), test_size=args.val_ratio, stratify=dataset.data['label'])
        train_dataset = torch.utils.data.Subset(dataset, train_ids)
        valid_dataset = torch.utils.data.Subset(dataset, valid_ids)

        if verbose:
            print("="*20)
            print("train-valid split to train:", len(train_dataset), "valid:", len(valid_dataset))
            print("train:")
            print(dataset.data['label'].iloc[train_ids].value_counts())
            print("test:")
            print(dataset.data['label'].iloc[valid_ids].value_counts())
            print("="*20)

    # Build DataLoader
    BATCH_SIZE = args.batch_size
    VAL_BATCH_SIZE = args.val_batch_size if args.val_batch_size else BATCH_SIZE
    MAX_PAD_LEN = args.max_pad_len

    # Train
    NUM_EPOCHS = args.epochs
    SAVE_EVERY = args.save_every
    EVAL_EVERY = args.eval_every
    LOG_EVERY = args.log_every

    LEARNING_RATE = args.lr
    LR_TYPE = args.lr_type
    DECAY_RATE = args.lr_weight_decay
    WARMUPS = args.lr_warmups

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,                        # output directory
        logging_dir=LOG_DIR,                        # directory for storing logs

        save_total_limit=5,                         # number of total models saved.
        save_steps=SAVE_EVERY,                      # model saving step.
        logging_steps=LOG_EVERY,                    # log saving step.
        eval_steps=EVAL_EVERY,                      # evaluation step.
        
        num_train_epochs=NUM_EPOCHS,                # total number of training epochs
        evaluation_strategy='steps',
        save_strategy='steps',                      # evaluation strategy to adopt during training
        # `no`   : No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        load_best_model_at_end=True,
        
        per_device_train_batch_size=BATCH_SIZE,     # batch size per device during training
        per_device_eval_batch_size=VAL_BATCH_SIZE,  # batch size for evaluation
        
        learning_rate=LEARNING_RATE,                # learning_rate
        lr_scheduler_type=LR_TYPE,               # linear, cosine, cosine_with_restarts, 
                                                    # polynomial, constant, constant_with_warmup
        warmup_steps=WARMUPS,                       # number of warmup steps for learning rate scheduler
        weight_decay=DECAY_RATE,                    # strength of weight decay
    )

    trainer = None
    if valid_dataset is not None:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,          # evaluation dataset
            compute_metrics=compute_metrics      # define metrics function
        )
    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,                  # training arguments, defined above
            train_dataset=dataset,               # training dataset
            eval_dataset=dataset,                # evaluate with the whole dataset
            compute_metrics=compute_metrics      # define metrics function
        )

    # train model
    trainer.train()
    model.save_pretrained(os.path.join(SAVE_DIR, args.name + "_final"))


def main():
    
    parser = argparse.ArgumentParser(
        description="Train the model with the arguments given")
    args = parse_arguments(parser)

    v = args.verbose == "y"

    if args.seed is not None:
        set_all_seeds(args.seed, verbose=v)

    train(args, verbose=v)

if __name__ == '__main__':
    main()
