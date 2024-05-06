#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from utils import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import random
import argparse
import datetime
import logging
import os, os.path
import shutil
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from transformers import AdamW, AutoTokenizer, RobertaConfig
from model_weighted_roberta import *
import wandb


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8
    """
    m = nn.Softmax(dim=1)
    probabilities = m(preds)
    values, indices = torch.max(probabilities, 1)
    y_pred = indices
    acc = accuracy_score(y, y_pred)
    return acc


def eval_metrics(preds, y, num_labels):
    '''
    Returns performance metrics of predictor
    :param y: ground truth label
    :param preds: predicted logits
    :param preds: predicted logits
    :return: auc, acc, tn, fp, fn, tp
    '''
    m = nn.Softmax(dim=1)
    probabilities = m(torch.tensor(preds))
    y_values, indices = torch.max(probabilities, 1)
    y_pred = indices
    if num_labels == 2:
        try:
            auc = roc_auc_score(y, y_values)
        except ValueError:
            auc = np.array(0)
        acc = accuracy_score(y, y_pred)

        conf_mat = confusion_matrix(y, y_pred, labels=[0, 1])
        tn = conf_mat[0, 0]
        fp = conf_mat[0, 1]
        fn = conf_mat[1, 0]
        tp = conf_mat[1, 1]
        return auc, acc, tn, fp, fn, tp
    else:
        acc = accuracy_score(y, y_pred)
        conf_mat = confusion_matrix(y, y_pred)
        return acc, conf_mat


def simple_tokenize(orig_tokens, tokenizer, orig_labels, label_map, max_seq_length):
    """
    tokenize a array of raw text
    """
    # orig_tokens = orig_tokens.split()

    pad_token_label_id = -100
    tokens = []
    label_ids = []
    for word, label in zip(orig_tokens, orig_labels):
        word_tokens = tokenizer.tokenize(word)

        # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
        if len(word_tokens) > 0:
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = tokenizer.num_special_tokens_to_add()
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]

    bert_tokens = [tokenizer.cls_token]
    # bert_tokens = ["[CLS]"]

    bert_tokens.extend(tokens)
    label_ids = [pad_token_label_id] + label_ids

    bert_tokens.append(tokenizer.sep_token)
    # bert_tokens.append("[SEP]")
    label_ids += [pad_token_label_id]

    return bert_tokens, label_ids


def tokenize_with_new_mask(orig_text, max_length, tokenizer, orig_labels, label_map):
    """
    tokenize a array of raw text and generate corresponding
    attention labels array and attention masks array
    """
    pad_token_label_id = -100
    simple_tokenize_results = [list(tt) for tt in zip(
        *[simple_tokenize(orig_text[i], tokenizer, orig_labels[i], label_map, max_length) for i in
          range(len(orig_text))])]
    bert_tokens, label_ids = simple_tokenize_results[0], simple_tokenize_results[1]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in bert_tokens]
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    label_ids = pad_sequences(label_ids, maxlen=max_length, dtype="long", truncating="post", padding="post",
                              value=pad_token_label_id)

    attention_masks = np.array([[float(i > 0) for i in seq] for seq in input_ids])
    return input_ids, attention_masks, label_ids


def simple_tokenize_with_attention(orig_tokens, tokenizer, orig_labels, label_map, max_seq_length, orig_atts):
    """
    tokenize a array of raw text
    """
    # orig_tokens = orig_tokens.split()

    pad_token_label_id = -100
    tokens = []
    label_ids = []
    att_labels = []
    for word, label, att in zip(orig_tokens, orig_labels, orig_atts):
        word_tokens = tokenizer.tokenize(word)

        # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
        if len(word_tokens) > 0:
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            att_labels.extend([att] * (len(word_tokens)))

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = tokenizer.num_special_tokens_to_add()
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]
        att_labels = att_labels[: (max_seq_length - special_tokens_count)]

    bert_tokens = [tokenizer.cls_token]
    # bert_tokens = ["[CLS]"]

    bert_tokens.extend(tokens)
    label_ids = [pad_token_label_id] + label_ids
    att_labels = [0] + att_labels

    bert_tokens.append(tokenizer.sep_token)
    # bert_tokens.append("[SEP]")
    label_ids += [pad_token_label_id]
    att_labels += [0]

    return bert_tokens, label_ids, att_labels


def tokenize_with_given_mask(orig_text, max_length, tokenizer, orig_labels, label_map, orig_att):
    """
    tokenize a array of raw text and generate corresponding
    attention labels array and attention masks array
    """
    pad_token_label_id = -100
    simple_tokenize_results = [list(tt) for tt in zip(
        *[simple_tokenize_with_attention(orig_text[i], tokenizer, orig_labels[i], label_map, max_length, orig_att[i])
          for i in
          range(len(orig_text))])]
    bert_tokens, label_ids, att_labels = simple_tokenize_results[0], simple_tokenize_results[1], \
                                         simple_tokenize_results[2]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in bert_tokens]
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    label_ids = pad_sequences(label_ids, maxlen=max_length, dtype="long", truncating="post", padding="post",
                              value=pad_token_label_id)
    att_labels = pad_sequences(att_labels, maxlen=max_length, dtype="long", truncating="post", padding="post",
                               value=0)

    attention_masks = np.array([[float(i > 0 and j > 0) for i, j in zip(seq, att)] \
                                for seq, att in zip(input_ids, att_labels)])
    return input_ids, attention_masks, label_ids


def get_embedding(text_list, embeddings_index, embeddings, max_length, token_label_raw_list, label_map):
    pad_token_label_id = -100
    output_embedding = []
    label_ids_list = []
    attention_masks_list = []
    for words, token_labels in zip(text_list, token_label_raw_list):
        words_mapped = [0] * max_length
        label_ids = [0] * max_length
        length = len(words)
        if length < max_length:
            for i in range(0, length):
                words_mapped[i] = embeddings_index.get(words[i], -1)
                label_ids[i] = label_map[token_labels[i]]
            for i in range(length, max_length):
                words_mapped[i] = -2
                label_ids[i] = pad_token_label_id
        elif length > max_length:
            print('We should never see this print either')
        else:
            for i in range(0, max_length):
                words_mapped[i] = embeddings_index.get(words[i], -1)
                label_ids[i] = label_map[token_labels[i]]

        output_embedding.append(np.array([embeddings[ix] for ix in words_mapped]))
        label_ids_list.append(label_ids)
        attention_masks_list.append([float(i >= 0) for i in label_ids])
    output_embedding = np.array(output_embedding)
    attention_masks_list = np.array(attention_masks_list)
    label_ids_list = np.array(label_ids_list)
    return output_embedding, attention_masks_list, label_ids_list


def get_embedding_with_att(text_list, embeddings_index, embeddings, max_length, token_label_raw_list, label_map,
                           att_list):
    pad_token_label_id = -100
    output_embedding = []
    label_ids_list = []
    attention_masks_list = []
    for words, token_labels, att in zip(text_list, token_label_raw_list, att_list):
        words_mapped = [0] * max_length
        label_ids = [0] * max_length
        attention_masks = [0] * max_length
        length = len(words)
        if length < max_length:
            for i in range(0, length):
                words_mapped[i] = embeddings_index.get(words[i], -1)
                label_ids[i] = label_map[token_labels[i]]
                attention_masks[i] = att[i]
            for i in range(length, max_length):
                words_mapped[i] = -2
                label_ids[i] = pad_token_label_id
                attention_masks[i] = 0
        elif length > max_length:
            print('We should never see this print either')
        else:
            for i in range(0, max_length):
                words_mapped[i] = embeddings_index.get(words[i], -1)
                label_ids[i] = label_map[token_labels[i]]
                attention_masks[i] = att[i]

        output_embedding.append(np.array([embeddings[ix] for ix in words_mapped]))
        label_ids_list.append(label_ids)
        attention_masks_list.append([float(i >= 0 and j > 0) for i, j in zip(label_ids, attention_masks)])
    output_embedding = np.array(output_embedding)
    attention_masks_list = np.array(attention_masks_list)
    label_ids_list = np.array(label_ids_list)
    return output_embedding, attention_masks_list, label_ids_list


def train(model, optimizer, train_batch_generator, num_batches, device, args, label_map, token_weight, y_weight, num_labels):
    """
    Main training routine
    """
    epoch_loss = 0
    t_pred_list, s_pred_list, t_label_list, s_label_list = [], [], [], []
    if_crf = type(model) in [RobertaForTokenAndSequenceClassificationWithCRF, BiLSTMForWeightedTokenAndSequenceClassificationWithCRF]

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        x_batch, y_batch_l, t_batch_l, masks_batch = next(train_batch_generator)
        if len(x_batch.shape) == 3:
            x_batch = Variable(torch.FloatTensor(x_batch)).to(device)
        else:
            x_batch = Variable(torch.LongTensor(x_batch)).to(device)
        y_batch_l = y_batch_l.astype(float)
        y_batch_l = torch.LongTensor(y_batch_l)
        y_batch = Variable(y_batch_l).to(device)
        t_batch_l = t_batch_l.astype(float)
        t_batch_l = torch.LongTensor(t_batch_l)
        t_batch = Variable(t_batch_l).to(device)
        masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
        token_weight = token_weight.to(device) if token_weight is not None else None
        y_weight = y_weight.to(device) if y_weight is not None else None
        optimizer.zero_grad()
        outputs = model(input_ids=x_batch, attention_mask=masks_batch,
                        seq_labels=y_batch, token_labels=t_batch,
                        token_class_weight=token_weight, seq_class_weight=y_weight,
                        token_lambda=args.token_lambda,)

        loss, token_logits, seq_logits = outputs[:3]

        loss.backward()
        optimizer.step()

        t_batch_l = t_batch.detach().cpu()
        if if_crf:
            t_batch_filtered = [t_batch_l[i][t_batch_l[i] >= 0].tolist() for i in range(t_batch_l.shape[0])]
            t_pred_list.append(outputs[3])
            t_label_list.append(t_batch_filtered)
        else:
            t_pred_list.append(token_logits.detach().cpu().numpy())
            t_label_list.append(t_batch_l)
        s_pred_list.append(seq_logits.detach().cpu().numpy())
        s_label_list.append(y_batch.detach().cpu().numpy())

    output_s_pred = np.concatenate(s_pred_list, axis=0)
    if if_crf:
        output_t_pred = sum(t_pred_list, [])
        t_label = sum(t_label_list, [])
    else:
        output_t_pred = np.concatenate(t_pred_list, axis=0)
        t_label = np.concatenate(t_label_list, axis=0)
    s_label = np.concatenate(s_label_list, axis=0)

    print(f'\tClassification Loss: {epoch_loss / num_batches:.3f}')
    if num_labels == 2:
        s_auc, s_acc, s_tn, s_fp, s_fn, s_tp = eval_metrics(output_s_pred, s_label, num_labels)
    else:
        s_acc, s_conf = eval_metrics(output_s_pred, s_label, num_labels)
    if if_crf:
        t_eval_metrics = compute_crf_metrics(output_t_pred, t_label, label_map)
    else:
        t_eval_metrics = compute_metrics(output_t_pred, t_label, label_map)

    t_acc = t_eval_metrics["accuracy_score"]
    t_results = t_eval_metrics["results"]
    t_results_by_tag = t_eval_metrics["results_by_tag"]
    t_CR = t_eval_metrics["CR"]

    if num_labels == 2:
        return_s_tuple = (epoch_loss / num_batches, s_auc, s_acc, s_tn, s_fp, s_fn, s_tp)
    else:
        return_s_tuple = (epoch_loss / num_batches, s_acc, s_conf)
    return_t_tuple = (t_acc, t_results, t_results_by_tag, t_CR)
    return_tuple = (return_s_tuple, return_t_tuple)
    return return_tuple


def evaluate(model, test_batch_generator, num_batches, device, args, label_map, token_weight, y_weight, num_labels):
    """
    Main evaluation routine
    """
    epoch_loss = 0
    t_pred_list, s_pred_list, t_label_list, s_label_list = [], [], [], []
    if_crf = type(model) in [RobertaForTokenAndSequenceClassificationWithCRF, BiLSTMForWeightedTokenAndSequenceClassificationWithCRF]

    model.eval()

    with torch.no_grad():
        for b in tqdm(range(num_batches)):
            x_batch, y_batch, t_batch, masks_batch = next(test_batch_generator)
            if len(x_batch.shape) == 3:
                x_batch = Variable(torch.FloatTensor(x_batch)).to(device)
            else:
                x_batch = Variable(torch.LongTensor(x_batch)).to(device)
            y_batch = y_batch.astype(float)
            y_batch = Variable(torch.LongTensor(y_batch)).to(device)
            t_batch = t_batch.astype(float)
            t_batch = Variable(torch.LongTensor(t_batch)).to(device)
            masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
            token_weight = token_weight.to(device) if token_weight is not None else None
            y_weight = y_weight.to(device) if y_weight is not None else None

            outputs = model(input_ids=x_batch, attention_mask=masks_batch,
                            seq_labels=y_batch, token_labels=t_batch,
                            token_class_weight=token_weight, seq_class_weight=y_weight,
                            token_lambda=args.token_lambda, )

            loss, token_logits, seq_logits = outputs[:3]

            epoch_loss += loss.item()

            t_batch_l = t_batch.detach().cpu()
            if if_crf:
                t_batch_filtered = [t_batch_l[i][t_batch_l[i] >= 0].tolist() for i in range(t_batch_l.shape[0])]
                t_pred_list.append(outputs[3])
                t_label_list.append(t_batch_filtered)
            else:
                t_pred_list.append(token_logits.detach().cpu().numpy())
                t_label_list.append(t_batch_l)
            s_pred_list.append(seq_logits.detach().cpu().numpy())
            s_label_list.append(y_batch.detach().cpu().numpy())

        output_s_pred = np.concatenate(s_pred_list, axis=0)
        if if_crf:
            output_t_pred = sum(t_pred_list, [])
            t_label = sum(t_label_list, [])
        else:
            output_t_pred = np.concatenate(t_pred_list, axis=0)
            t_label = np.concatenate(t_label_list, axis=0)
        s_label = np.concatenate(s_label_list, axis=0)

        print(f'\tClassification Loss: {epoch_loss / num_batches:.3f}')
        if num_labels == 2:
            s_auc, s_acc, s_tn, s_fp, s_fn, s_tp = eval_metrics(output_s_pred, s_label, num_labels)
        else:
            s_acc, s_conf = eval_metrics(output_s_pred, s_label, num_labels)
        if if_crf:
            t_eval_metrics = compute_crf_metrics(output_t_pred, t_label, label_map)
        else:
            t_eval_metrics = compute_metrics(output_t_pred, t_label, label_map)

        t_acc = t_eval_metrics["accuracy_score"]
        t_results = t_eval_metrics["results"]
        t_results_by_tag = t_eval_metrics["results_by_tag"]
        t_CR = t_eval_metrics["CR"]

        if num_labels == 2:
            return_s_tuple = (epoch_loss / num_batches, s_auc, s_acc, s_tn, s_fp, s_fn, s_tp)
        else:
            return_s_tuple = (epoch_loss / num_batches, s_acc, s_conf)
        return_t_tuple = (t_acc, t_results, t_results_by_tag, t_CR)
        return_tuple = (return_s_tuple, return_t_tuple, output_t_pred, output_s_pred)
        return return_tuple


def load_model(model_type, model_path, config):
    roberta_class = {'bertweet-multi': RobertaForTokenAndSequenceClassification,
                     'bertweet-multi-crf': RobertaForTokenAndSequenceClassificationWithCRF}
    bilstm_class = {'BiLSTM-multi': BiLSTMForWeightedTokenAndSequenceClassification,
                    'BiLSTM-multi-crf': BiLSTMForWeightedTokenAndSequenceClassificationWithCRF,
                    'BiLSTM-multi-ver2': BiLSTMForWeightedTokenAndSequenceClassificationVer2,
                    'BiLSTM-multi-crf-ver2': BiLSTMForWeightedTokenAndSequenceClassificationWithCRFVer2,
                    'BiLSTM-multi-ver3': BiLSTMForWeightedTokenAndSequenceClassificationVer3,
                    'BiLSTM-multi-crf-ver3': BiLSTMForWeightedTokenAndSequenceClassificationWithCRFVer3,
                    'BiLSTM-multi-ver4': BiLSTMForWeightedTokenAndSequenceClassificationVer4,
                    'BiLSTM-multi-crf-ver4': BiLSTMForWeightedTokenAndSequenceClassificationWithCRFVer4,
                    }
    if model_type in roberta_class.keys():
        model = roberta_class[model_type].from_pretrained(model_path, config=config)
    elif model_type in bilstm_class.keys():
        model = bilstm_class[model_type](config=config)
        if model_path is not None:
            model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.pt')))
    else:
        model = None
    return model


NOTE = 'V1.0.0: Initial Public Version'


### Main
def main():
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument("--bert_model", default=None, type=str)
    parser.add_argument("--model_type", default=None, type=str)
    parser.add_argument("--task_type", default='entity_detection', type=str)
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--rnn_hidden_size', default=384, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=300, type=int)
    parser.add_argument('--test_batch_size', default=300, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--data', default='wnut_16', type=str)
    parser.add_argument('--log_dir', default='log-BERTweet-multi', type=str)
    parser.add_argument("--save_model", default=False, action='store_true')
    parser.add_argument("--early_stop", default=False, action='store_true')
    parser.add_argument("--freeze_bert", default=False, action='store_true')
    parser.add_argument("--miss_token_label", default=False, action='store_true')
    parser.add_argument("--extra_real_label", default=False, action='store_true')
    parser.add_argument("--fixed_epoch", default=False, action='store_true')
    parser.add_argument("--assign_token_weight", default=False, type='bool')
    parser.add_argument("--assign_seq_weight", default=False, type='bool')
    parser.add_argument('--token_lambda', default=10, type=float)
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--val_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--label_map", default=None, type=str)
    parser.add_argument("--performance_file", default='all_test_performance.txt', type=str)
    parser.add_argument("--embeddings_file", default='glove.840B.300d.txt', type=str)

    args = parser.parse_args()

    assert args.task_type in ['entity_detection', 'relevant_entity_detection', 'entity_relevance_classification']

    print("cuda is available:", torch.cuda.is_available())
    log_directory = args.log_dir + '/' + str(args.bert_model).split('/')[-1] + '/' + args.model_type + '/' + \
                    args.task_type + '/' + str(args.n_epochs) + '_epoch/' + args.data.split('/')[-1] + '/' + \
                    str(args.train_file).replace('/', '_') + '_' + str(args.val_file).replace('/', '_') + '_' + \
                    str(args.test_file).replace('/', '_') + '/' + \
                    str(args.assign_token_weight) + '_token_weight/' + str(args.assign_seq_weight) + '_seq_weight/' + \
                    str(args.freeze_bert) + '_freeze_bert/' + str(args.miss_token_label) + '_miss_token_label/' + \
                    str(args.extra_real_label) + '_extra_real_label/' + str(args.fixed_epoch) + '_fixed_epoch/' + \
                    str(args.token_lambda) + '_token_lambda/' + str(args.seed) + '_seed/'
    log_filename = 'log.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.', '-') + '.txt'
    model_dir = 'saved-model'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logname = os.path.join(log_directory, log_filename)
    modeldir = os.path.join(log_directory, model_dir)
    wandb.init(
        # set the wandb project where this run will be logged
        project="Dissertation-Task-3",
        name=f"{args.model_type}-{args.bert_model}-{args.task_type}-{args.train_file}-{args.seed}",
        # track hyperparameters and run metadata
        config=args
    )
    wandb.run.name = wandb.run.name + '-' + wandb.run.id

    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    if os.path.exists(modeldir) and os.listdir(modeldir):
        logging.info(f"modeldir {modeldir} already exists and it is not empty")
        print(f"modeldir {modeldir} already exists and it is not empty")
    else:
        os.makedirs(modeldir, exist_ok=True)
        logging.info(f"Create modeldir: {modeldir}")
        print(f"Create modeldir: {modeldir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    train_data = pd.read_pickle(os.path.join(args.data, args.train_file))
    val_data = pd.read_pickle(os.path.join(args.data, args.val_file))
    test_data = pd.read_pickle(os.path.join(args.data, args.test_file))
    need_columns = ['tweet_tokens']
    if args.task_type == 'entity_detection':
        need_columns.append('entity_label')
    elif args.task_type == 'relevant_entity_detection':
        need_columns.append('relevant_entity_label')
    elif args.task_type == 'entity_relevance_classification':
        need_columns.append('relevance_entity_class_label')
    need_columns.append('sentence_class')
    if args.miss_token_label is True:
        need_columns.append('att_label')
        if args.extra_real_label is True:
            need_columns.append('real_relevant_entity_label')
            X_train_raw, token_label_train_raw, Y_train, att_label_train_raw, real_token_label_train_raw = extract_from_dataframe(train_data,
                                                                                                  need_columns)
            X_dev_raw, token_label_dev_raw, Y_dev, att_label_dev_raw, real_token_label_dev_raw = extract_from_dataframe(val_data, need_columns)
            X_test_raw, token_label_test_raw, Y_test = extract_from_dataframe(test_data, need_columns[:-2])
        else:
            X_train_raw, token_label_train_raw, Y_train, att_label_train_raw = extract_from_dataframe(train_data,
                                                                                                      need_columns)
            X_dev_raw, token_label_dev_raw, Y_dev, att_label_dev_raw = extract_from_dataframe(val_data, need_columns)
            X_test_raw, token_label_test_raw, Y_test = extract_from_dataframe(test_data, need_columns[:-1])
    else:
        if args.extra_real_label is True:
            need_columns.append('real_relevant_entity_label')
            X_train_raw, token_label_train_raw, Y_train, real_token_label_train_raw = extract_from_dataframe(train_data, need_columns)
            X_dev_raw, token_label_dev_raw, Y_dev, real_token_label_dev_raw = extract_from_dataframe(val_data, need_columns)
            X_test_raw, token_label_test_raw, Y_test = extract_from_dataframe(test_data, need_columns[:-1])
        else:
            X_train_raw, token_label_train_raw, Y_train = extract_from_dataframe(train_data, need_columns)
            X_dev_raw, token_label_dev_raw, Y_dev = extract_from_dataframe(val_data, need_columns)
            X_test_raw, token_label_test_raw, Y_test = extract_from_dataframe(test_data, need_columns)
    args.eval_batch_size = Y_dev.shape[0]
    args.test_batch_size = Y_test.shape[0]

    with open(os.path.join(args.data, args.label_map), 'r') as fp:
        label_map = json.load(fp)

    labels = list(label_map.keys())

    logging.info(args)
    print(args)

    if args.bert_model is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model, normalization=True)

        if args.extra_real_label is True:
            _, _, real_token_label_train = tokenize_with_new_mask(
                X_train_raw, args.max_length, tokenizer, real_token_label_train_raw, label_map)
            _, _, real_token_label_dev = tokenize_with_new_mask(
                X_dev_raw, args.max_length, tokenizer, real_token_label_dev_raw, label_map)

        if args.miss_token_label is True:
            X_train, masks_train, token_label_train = tokenize_with_given_mask(
                X_train_raw, args.max_length, tokenizer, token_label_train_raw, label_map, att_label_train_raw)
            X_dev, masks_dev, token_label_dev = tokenize_with_given_mask(
                X_dev_raw, args.max_length, tokenizer, token_label_dev_raw, label_map, att_label_dev_raw)
            X_test, masks_test, token_label_test = tokenize_with_new_mask(
                X_test_raw, args.max_length, tokenizer, token_label_test_raw, label_map)
        else:
            X_train, masks_train, token_label_train = tokenize_with_new_mask(
                X_train_raw, args.max_length, tokenizer, token_label_train_raw, label_map)
            X_dev, masks_dev, token_label_dev = tokenize_with_new_mask(
                X_dev_raw, args.max_length, tokenizer, token_label_dev_raw, label_map)
            X_test, masks_test, token_label_test = tokenize_with_new_mask(
                X_test_raw, args.max_length, tokenizer, token_label_test_raw, label_map)
    else:
        embeddings_index, embeddings = new_build_glove_embedding(
            embedding_path=args.embeddings_file)
        if args.extra_real_label is True:
            _, _, real_token_label_train = get_embedding(X_train_raw, embeddings_index, embeddings,
                                                         args.max_length,
                                                         real_token_label_train_raw, label_map)
            _, _, real_token_label_dev = get_embedding(X_dev_raw, embeddings_index, embeddings,
                                                       args.max_length,
                                                       real_token_label_dev_raw, label_map)
        if args.miss_token_label is True:
            X_train, masks_train, token_label_train = get_embedding_with_att(X_train_raw, embeddings_index, embeddings,
                                                                             args.max_length,
                                                                             token_label_train_raw, label_map,
                                                                             att_label_train_raw)
            X_dev, masks_dev, token_label_dev = get_embedding_with_att(X_dev_raw, embeddings_index, embeddings,
                                                                       args.max_length,
                                                                       token_label_dev_raw, label_map,
                                                                       att_label_dev_raw)
            X_test, masks_test, token_label_test = get_embedding(X_test_raw, embeddings_index, embeddings,
                                                                 args.max_length,
                                                                 token_label_test_raw, label_map)
        else:
            X_train, masks_train, token_label_train = get_embedding(X_train_raw, embeddings_index, embeddings,
                                                                    args.max_length,
                                                                    token_label_train_raw, label_map)
            X_dev, masks_dev, token_label_dev = get_embedding(X_dev_raw, embeddings_index, embeddings,
                                                              args.max_length,
                                                              token_label_dev_raw, label_map)
            X_test, masks_test, token_label_test = get_embedding(X_test_raw, embeddings_index, embeddings,
                                                                 args.max_length,
                                                                 token_label_test_raw, label_map)

    num_labels = len(set(np.concatenate([Y_train, Y_dev, Y_test])))
    # weight of each class in loss function
    token_weight = None
    if args.assign_token_weight:
        token_weight = [token_label_train.shape[0] / (token_label_train == i).sum() for i in range(len(labels))]
        token_weight = torch.FloatTensor(token_weight)

    y_weight = None
    if args.assign_seq_weight:
        y_weight = [np.array(Y_train).shape[0] / (np.array(Y_train) == i).sum() for i in range(num_labels)]
        y_weight = torch.FloatTensor(y_weight)

    if args.bert_model is not None:
        config = RobertaConfig.from_pretrained(args.bert_model)
        config.update({'num_token_labels': len(labels), 'num_labels': num_labels,
                       'token_label_map': label_map, })
    else:
        from dotmap import DotMap
        config = DotMap()
        config.update({'num_token_labels': len(labels), 'num_labels': num_labels,
                       'token_label_map': label_map, 'hidden_dropout_prob': 0.1,
                       'rnn_hidden_dimension': args.rnn_hidden_size})

    model = load_model(args.model_type, args.bert_model, config)
    binary_task = num_labels == 2

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    param_optimizer = list(model.named_parameters())
    if args.freeze_bert:
        for n, p in param_optimizer:
            if n.startswith('roberta'):
                p.requires_grad = False

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    model = model.to(device)

    best_valid_s_auc, best_valid_s_acc = 0, 0

    best_valid_t_acc, best_valid_t_F = 0, 0
    best_valid_s_tuple, best_valid_t_tuple = None, None
    train_losses, eval_losses, r_train_losses, r_eval_losses = [], [], [], []
    train_s_acc_list, eval_s_acc_list, r_train_s_acc_list, r_eval_s_acc_list = [], [], [], []
    train_t_F_list, eval_t_F_list, r_train_t_F_list, r_eval_t_F_list = [], [], [], []

    early_stop_sign = 0
    for epoch in range(args.n_epochs):

        # train

        train_batch_generator = multi_batch_generator(X_train, Y_train, token_label_train,
                                                      masks_train, args.batch_size)
        num_batches = X_train.shape[0] // args.batch_size
        train_outputs = train(model, optimizer, train_batch_generator, num_batches, device,
                              args, label_map, token_weight, y_weight, num_labels)
        train_batch_generator = multi_batch_seq_generator(X_train, Y_train, token_label_train, masks_train, args.batch_size)
        num_batches = X_train.shape[0] // args.batch_size
        train_outputs = evaluate(model, train_batch_generator, num_batches, device,
                                 args, label_map, token_weight, y_weight, num_labels)

        train_s_tuple, train_t_tuple = train_outputs[:2]
        if len(train_outputs) > 2:
            train_bt_tuple = train_outputs[2]
        train_losses.append(train_s_tuple[0])
        train_s_acc = train_s_tuple[2] if binary_task else train_s_tuple[1]
        train_t_F = train_t_tuple[1]['strict']['f1']
        train_s_acc_list.append(train_s_acc)
        train_t_F_list.append(train_t_F)
        

        # eval
        dev_batch_generator = multi_batch_seq_generator(X_dev, Y_dev, token_label_dev, masks_dev,
                                                        min(X_dev.shape[0], args.eval_batch_size))
        num_batches = X_dev.shape[0] // min(X_dev.shape[0], args.eval_batch_size)
        valid_outputs = evaluate(model, dev_batch_generator, num_batches, device,
                                 args, label_map, token_weight, y_weight, num_labels)
        valid_s_tuple, valid_t_tuple, valid_t_pred, valid_s_pred = valid_outputs[:4]

        eval_losses.append(valid_s_tuple[0])
        valid_s_acc = valid_s_tuple[2] if binary_task else valid_s_tuple[1]
        valid_t_F = valid_t_tuple[1]['strict']['f1']
        eval_s_acc_list.append(valid_s_acc)
        eval_t_F_list.append(valid_t_F)

        good_cond_s = best_valid_s_acc < valid_s_acc
        normal_cond_s = (abs(best_valid_s_acc - valid_s_acc) < 0.03) or good_cond_s
        good_cond_t = best_valid_t_F < valid_t_F
        normal_cond_t = abs(best_valid_t_F - valid_t_F) < 0.03 or good_cond_t
        if binary_task:
            train_s_dict = {'train_seq_'+key: val for key, val in zip(['loss', 'auc', 'acc', 'tn', 'fp', 'fn', 'tp'], train_s_tuple)}
            valid_s_dict = {'valid_seq_'+key: val for key, val in zip(['loss', 'auc', 'acc', 'tn', 'fp', 'fn', 'tp'], valid_s_tuple)}
        else:
            train_s_dict = {'train_seq_'+key: val for key, val in zip(['loss', 'acc', 'conf'], train_s_tuple)}
            valid_s_dict = {'valid_seq_'+key: val for key, val in zip(['loss', 'acc', 'conf'], valid_s_tuple)}
        train_t_dict = {'train_token_'+key: val for key, val in zip(['acc', 'results', 'results_by_tag', 'CR'], train_t_tuple)}
        valid_t_dict = {'valid_token_'+key: val for key, val in zip(['acc', 'results', 'results_by_tag', 'CR'], valid_t_tuple)}
        
        r_train_s_dict, r_valid_s_dict, r_train_t_dict, r_valid_t_dict = {}, {}, {}, {}
        # eval on real labels
        if args.extra_real_label is True:
            r_train_batch_generator = multi_batch_seq_generator(X_train, Y_train, real_token_label_train,
                                                                masks_train, args.batch_size)
            num_batches = X_train.shape[0] // args.batch_size
            r_train_outputs = evaluate(model, r_train_batch_generator, num_batches, device,
                                       args, label_map, token_weight, y_weight, num_labels)
            r_train_s_tuple, r_train_t_tuple, r_train_t_pred, r_train_s_pred = r_train_outputs[:4]

            r_train_losses.append(r_train_s_tuple[0])
            r_train_s_acc = r_train_s_tuple[2] if binary_task else r_train_s_tuple[1]
            r_train_t_F = r_train_t_tuple[1]['strict']['f1']
            r_train_s_acc_list.append(r_train_s_acc)
            r_train_t_F_list.append(r_train_t_F)

            r_dev_batch_generator = multi_batch_seq_generator(X_dev, Y_dev, real_token_label_dev, masks_dev,
                                                            min(X_dev.shape[0], args.eval_batch_size))
            num_batches = X_dev.shape[0] // min(X_dev.shape[0], args.eval_batch_size)
            r_valid_outputs = evaluate(model, r_dev_batch_generator, num_batches, device,
                                     args, label_map, token_weight, y_weight, num_labels)
            r_valid_s_tuple, r_valid_t_tuple, r_valid_t_pred, r_valid_s_pred = r_valid_outputs[:4]

            r_eval_losses.append(r_valid_s_tuple[0])
            r_valid_s_acc = r_valid_s_tuple[2] if binary_task else r_valid_s_tuple[1]
            r_valid_t_F = r_valid_t_tuple[1]['strict']['f1']
            r_eval_s_acc_list.append(r_valid_s_acc)
            r_eval_t_F_list.append(r_valid_t_F)
            if binary_task:
                r_train_s_dict = {'real_train_seq_'+key: val for key, val in zip(['loss', 'auc', 'acc', 'tn', 'fp', 'fn', 'tp'], r_train_s_tuple)}
                r_valid_s_dict = {'real_valid_seq_'+key: val for key, val in zip(['loss', 'auc', 'acc', 'tn', 'fp', 'fn', 'tp'], r_valid_s_tuple)}
            else:
                r_train_s_dict = {'real_train_seq_'+key: val for key, val in zip(['loss', 'acc', 'conf'], r_train_s_tuple)}
                r_valid_s_dict = {'real_valid_seq_'+key: val for key, val in zip(['loss', 'acc', 'conf'], r_valid_s_tuple)}
            r_train_t_dict = {'real_train_token_'+key: val for key, val in zip(['acc', 'results', 'results_by_tag', 'CR'], r_train_t_tuple)}
            r_valid_t_dict = {'real_valid_token_'+key: val for key, val in zip(['acc', 'results', 'results_by_tag', 'CR'], r_valid_t_tuple)}
        epoch_dict = {'epoch':epoch}
        for pdict in [train_s_dict, train_t_dict, valid_s_dict, valid_t_dict, r_train_s_dict, r_train_t_dict, r_valid_s_dict, r_valid_t_dict]:
            epoch_dict.update(pdict)
        wandb.log(epoch_dict)

        if (args.fixed_epoch and epoch == args.n_epochs - 1) or \
                ((not args.fixed_epoch) and ((good_cond_s and normal_cond_t) or (good_cond_t and normal_cond_s))) or (epoch == 0):
            if binary_task is True:
                best_valid_s_auc, best_valid_s_acc, best_valid_s_tn, best_valid_s_fp, best_valid_s_fn, best_valid_s_tp = valid_s_tuple[
                                                                                                                     1:]
            else:
                best_valid_s_acc = valid_s_tuple[1]
            best_valid_t_F = valid_t_F
            best_valid_s_tuple, best_valid_t_tuple = valid_s_tuple, valid_t_tuple
            best_train_t_tuple, best_train_s_tuple = train_t_tuple, train_s_tuple
            if args.extra_real_label is True:
                r_best_valid_s_tuple, r_best_valid_t_tuple = r_valid_s_tuple, r_valid_t_tuple
                r_best_train_t_tuple, r_best_train_s_tuple = r_train_t_tuple, r_train_s_tuple

            if args.bert_model is not None:
                model.save_pretrained(modeldir)
            else:
                torch.save(model.state_dict(), os.path.join(modeldir, 'pytorch_model.pt'))

            if (not args.fixed_epoch) and ((good_cond_s and normal_cond_t) or (good_cond_t and normal_cond_s) or (epoch == 0)):
                if args.early_stop:
                    early_stop_sign = 0
        elif (not args.fixed_epoch) and args.early_stop:
            early_stop_sign += 1

        content = f'Train Seq Acc: {train_s_acc * 100:.2f}%, Token F1: {train_t_F * 100:.2f}%. ' \
                  f'Val Seq Acc: {valid_s_acc * 100:.2f}%, Token F1: {valid_t_F * 100:.2f}%'
        print(content)
        logging.info(content)
        if (not args.fixed_epoch) and args.early_stop and early_stop_sign >= 5:
            break

    content = f"After {epoch + 1} epoch, Best valid token F1: {best_valid_t_F}, seq accuracy: {best_valid_s_acc}"
    print(content)
    logging.info(content)

    performance_dict = vars(args)
    if binary_task is True:
        performance_dict['S_best_train_AUC'], performance_dict['S_best_train_ACC'], \
        performance_dict['S_best_train_TN'], performance_dict['S_best_train_FP'], \
        performance_dict['S_best_train_FN'], performance_dict['S_best_train_TP'] = best_train_s_tuple[1:]

        performance_dict['S_best_valid_AUC'], performance_dict['S_best_valid_ACC'], \
        performance_dict['S_best_valid_TN'], performance_dict['S_best_valid_FP'], \
        performance_dict['S_best_valid_FN'], performance_dict['S_best_valid_TP'] = best_valid_s_tuple[1:]
    else:
        performance_dict['S_best_train_ACC'], performance_dict['S_best_train_CF'] = best_train_s_tuple[1:]
        performance_dict['S_best_valid_ACC'], performance_dict['S_best_valid_CF'] = best_valid_s_tuple[1:]

    performance_dict['T_best_train_ACC'], performance_dict['T_best_train_results'], \
    performance_dict['T_best_train_results_by_tag'], performance_dict['T_best_train_CR'] = best_train_t_tuple

    performance_dict['T_best_valid_ACC'], performance_dict['T_best_valid_results'], \
    performance_dict['T_best_valid_results_by_tag'], performance_dict['T_best_valid_CR'] = best_valid_t_tuple

    if args.extra_real_label is True:
        if binary_task is True:
            performance_dict['S_r_best_train_AUC'], performance_dict['S_r_best_train_ACC'], \
            performance_dict['S_r_best_train_TN'], performance_dict['S_r_best_train_FP'], \
            performance_dict['S_r_best_train_FN'], performance_dict['S_r_best_train_TP'] = r_best_train_s_tuple[1:]

            performance_dict['S_r_best_valid_AUC'], performance_dict['S_r_best_valid_ACC'], \
            performance_dict['S_r_best_valid_TN'], performance_dict['S_r_best_valid_FP'], \
            performance_dict['S_r_best_valid_FN'], performance_dict['S_r_best_valid_TP'] = r_best_valid_s_tuple[1:]
        else:
            performance_dict['S_r_best_train_ACC'], performance_dict['S_r_best_train_CF'] = r_best_train_s_tuple[1:]
            performance_dict['S_r_best_valid_ACC'], performance_dict['S_r_best_valid_CF'] = r_best_valid_s_tuple[1:]

        performance_dict['T_r_best_train_ACC'], performance_dict['T_r_best_train_results'], \
        performance_dict['T_r_best_train_results_by_tag'], performance_dict['T_r_best_train_CR'] = r_best_train_t_tuple

        performance_dict['T_r_best_valid_ACC'], performance_dict['T_r_best_valid_results'], \
        performance_dict['T_r_best_valid_results_by_tag'], performance_dict['T_r_best_valid_CR'] = r_best_valid_t_tuple

    # Plot training classification loss
    epoch_count = np.arange(1, epoch + 2)
    fig, axs = plt.subplots(3, figsize=(10, 18), sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
    axs[0].plot(epoch_count, train_losses, 'b--')
    axs[0].plot(epoch_count, eval_losses, 'g-')
    axs[0].legend(['Training Loss', 'Valid Loss'], fontsize=14)
    if args.extra_real_label is True:
        axs[0].plot(epoch_count, r_train_losses, 'r--')
        axs[0].plot(epoch_count, r_eval_losses, 'm-')
        axs[0].legend(['Training Loss', 'Valid Loss', 'Real Training Loss', 'Real Valid Loss'], fontsize=14)
    axs[0].set_ylabel('Loss', fontsize=16)
    axs[0].tick_params(axis='y', labelsize=14, labelcolor='b')
    axs[0].tick_params(axis='x', labelsize=14)

    axs[1].plot(epoch_count, train_t_F_list, 'b--')
    axs[1].plot(epoch_count, eval_t_F_list, 'g-')
    axs[1].legend(['Training Token F1', 'Valid Token F1'], fontsize=14)
    if args.extra_real_label is True:
        axs[1].plot(epoch_count, r_train_t_F_list, 'r--')
        axs[1].plot(epoch_count, r_eval_t_F_list, 'm-')
        axs[1].legend(['Training Token F1', 'Valid Token F1', 'Real Training Token F1', 'Real Valid Token F1'], fontsize=14)
    axs[1].set_ylabel('F1', fontsize=16)
    axs[1].tick_params(axis='y', labelsize=14, labelcolor='r')
    axs[1].tick_params(axis='x', labelsize=14)

    axs[2].plot(epoch_count, train_s_acc_list, 'b--')
    axs[2].plot(epoch_count, eval_s_acc_list, 'g-')
    axs[2].legend(['Training Seq Acc', 'Valid Seq Acc'], fontsize=14)
    if args.extra_real_label is True:
        axs[2].plot(epoch_count, r_train_s_acc_list, 'r--')
        axs[2].plot(epoch_count, r_eval_s_acc_list, 'm-')
        axs[2].legend(['Training Seq Acc', 'Valid Seq Acc', 'Real Training Seq Acc', 'Real Valid Seq Acc'],
                      fontsize=14)
    axs[2].set_ylabel('Acc', fontsize=16)
    axs[2].set_xlabel('Epoch', fontsize=16)
    axs[2].tick_params(axis='y', labelsize=14, labelcolor='y')
    axs[2].tick_params(axis='x', labelsize=14)

    figure_filename = 'fig.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.',
                                                                                                         '-') + '.png'
    figfullname = log_directory + figure_filename
    plt.savefig(figfullname, dpi=fig.dpi)

    num_batches = X_test.shape[0] // args.test_batch_size
    test_batch_generator = multi_batch_seq_generator(X_test, Y_test, token_label_test, masks_test,
                                                     args.test_batch_size)
    del model
    torch.cuda.empty_cache()
    model = load_model(args.model_type, modeldir, config)
    model = model.to(device)
    test_outputs = evaluate(model, test_batch_generator, num_batches, device,
                            args, label_map, token_weight, y_weight, num_labels)
    test_s_tuple, test_t_tuple, test_t_pred, test_s_pred = test_outputs[0:4]
    test_t_F = test_t_tuple[1]['strict']['f1']
    test_s_acc = test_s_tuple[2] if binary_task else test_s_tuple[1]
    content = f'Test Seq Acc: {test_s_acc * 100:.2f}%, Token F1: {test_t_F * 100:.2f}%'
    print(content)
    logging.info(content)
    token_pred_dir = log_directory + 'token_prediction.npy'
    if_crf = "crf" in args.model_type
    if if_crf:
        with open(token_pred_dir, "wb") as fp:
            pickle.dump(test_t_pred, fp)
    else:
        np.save(token_pred_dir, test_t_pred)
    seq_pred_dir = log_directory + 'seq_prediction.npy'
    np.save(seq_pred_dir, test_s_pred)

    if binary_task is True:
        performance_dict['S_best_test_AUC'], performance_dict['S_best_test_ACC'], \
        performance_dict['S_best_test_TN'], performance_dict['S_best_test_FP'], \
        performance_dict['S_best_test_FN'], performance_dict['S_best_test_TP'] = test_s_tuple[1:]
    else:
        performance_dict['S_best_test_ACC'], performance_dict['S_best_test_CF'] = test_s_tuple[1:]

    performance_dict['T_best_test_ACC'], performance_dict['T_best_test_results'], \
    performance_dict['T_best_test_results_by_tag'], performance_dict['T_best_test_CR'] = test_t_tuple

    performance_dict['script_file'] = os.path.basename(__file__)
    performance_dict['log_directory'] = log_directory
    performance_dict['log_filename'] = log_filename
    performance_dict['note'] = NOTE
    performance_dict['Time'] = str(datetime.datetime.now())
    performance_dict['device'] = torch.cuda.get_device_name(device) if device.type != 'cpu' else 'cpu'
    if binary_task:
        test_s_dict = {'test_seq_'+key: val for key, val in zip(['loss', 'auc', 'acc', 'tn', 'fp', 'fn', 'tp'], test_s_tuple)}
    else:
        test_s_dict = {'test_seq_'+key: val for key, val in zip(['loss', 'acc', 'conf'], test_s_tuple)}
    test_t_dict = {'test_token_'+key: val for key, val in zip(['acc', 'results', 'results_by_tag', 'CR'], test_t_tuple)}
    test_dict = test_s_dict
    test_dict.update(test_t_dict)
    wandb.log(test_dict)
    wandb.log({k:performance_dict[k] for k in ['script_file', 'log_directory', 'log_filename', 'note', 'Time', 'device']})
    
    for key, value in performance_dict.items():
        if type(value) is np.int64:
            performance_dict[key] = int(value)
        if type(value) is np.ndarray:
            performance_dict[key] = value.tolist()
    with open(args.performance_file, 'a+') as outfile:
        outfile.write(json.dumps(performance_dict) + '\n')
    if not args.save_model:
        shutil.rmtree(modeldir)


if __name__ == '__main__':
    main()
