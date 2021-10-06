from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from importlib import import_module
from tqdm import tqdm

from dataset.preprocessor.preprocessors import *

def inference(model, tokenized_sent, device, is_roberta='n'):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=1, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            if is_roberta == 'y':
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device)
                    )                
            else:
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    token_type_ids=data['token_type_ids'].to(device)
                    )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def main(args):
    """
        주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    Tokenizer_NAME = args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    # Build Test Dataset
    try:
        dataset_module = getattr(import_module(
            "dataset."+args.dataset), args.dataset)
    except:
        dataset_module = getattr(import_module(
            "dataset.dataset"), args.dataset)
  
    dataset = dataset_module(
        data_dir=args.data_dir,
        max_length=args.max_seq_len,
        num_labels=args.num_labels,
        dropna=True,
        is_test=True)
    # dataset must return
    # dict of {'input_ids', 'token_type_ids', 'attention_mask', 'labels'}
    # to work properly

    preprocessor = None
    if args.preprocessor:
        try:
            preprocessor_module = getattr(import_module(
                "dataset.preprocessor."+args.preprocessor), args.preprocessor)
        except:
            preprocessor_module = getattr(import_module(
                "dataset.preprocessor.preprocessors"), args.preprocessor)

        preprocessor = preprocessor_module()

    if preprocessor is not None:
        dataset.set_preprocessor(preprocessor)
    
    dataset.set_tokenizer(tokenizer)
    dataset.preprocess()

    test_id = dataset.get_id_column()

    added_token_num = dataset.get_special_token_num()

    ## load my model
    MODEL_NAME = args.model_dir
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    if added_token_num > 0:
        model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
    model.to(device)

    ## predict answer
    pred_answer, output_prob = inference(model, dataset, device, is_roberta=args.is_roberta)
    pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.

    ## make csv file with predicted answer
    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
    output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument('--model_dir', type=str, default="./saved/exp/exp_final")
    parser.add_argument('--is_roberta', type=str, default="n", help="model is roberta or not (y,n)")
    parser.add_argument('--tokenizer', type=str, default="klue/bert-base")
    parser.add_argument('--dataset', type=str, default="BaselineDataset")
    parser.add_argument('--preprocessor', type=str, default='BaselinePreprocessor', help="type of preprocessor (default: BaselinePreprocessor)")        
    parser.add_argument('--data_dir', type=str, default="/opt/ml/dataset")
    parser.add_argument('--max_seq_len', type=int, metavar='L', default=256, help="max sequence length (default 256)")
    parser.add_argument('--num_labels', type=int, default=30, help="number of labels for classification (default: 30)")                        

    args = parser.parse_args()
    #print(args)
    main(args)