#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from utils import *
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import argparse
import datetime
import logging
import os, os.path
import shutil
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from seqeval.metrics import accuracy_score
from transformers import AdamW, AutoTokenizer, AutoConfig, RobertaConfig
from model_weighted_roberta import *
import json
import wandb


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


def train(model, optimizer, train_batch_generator, num_batches, device, args, label_map, class_weight):
    """
    Main training routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_results = {}
    epoch_results_by_tag = {}
    epoch_CR = ""

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        x_batch, y_batch, masks_batch = next(train_batch_generator)
        if len(x_batch.shape) == 3:
            x_batch = Variable(torch.FloatTensor(x_batch)).to(device)
        else:
            x_batch = Variable(torch.LongTensor(x_batch)).to(device)
        y_batch = y_batch.astype(float)
        y_batch = Variable(torch.LongTensor(y_batch)).to(device)
        masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
        class_weight = class_weight.to(device) if class_weight is not None else None
        optimizer.zero_grad()

        outputs = model(input_ids=x_batch, attention_mask=masks_batch, labels=y_batch, class_weight=class_weight)

        loss, logits = outputs[:2]

        loss.backward()
        optimizer.step()

        if type(model) in [RobertaForTokenClassificationWithCRF, BiLSTMForTokenClassificationWithCRF]:
            y_batch = y_batch.detach().cpu()
            y_batch_filtered = [y_batch[i][y_batch[i] >= 0].tolist() for i in range(y_batch.shape[0])]
            eval_metrics = compute_crf_metrics(outputs[2], y_batch_filtered, label_map)
        else:
            eval_metrics = compute_metrics(logits.detach().cpu(), y_batch.detach().cpu(), label_map)

        epoch_loss += loss.item()
        epoch_acc += eval_metrics["accuracy_score"]
        epoch_results.update(eval_metrics["results"])
        epoch_results_by_tag.update(eval_metrics["results_by_tag"])
        epoch_CR = eval_metrics["CR"]
    print(f'\tClassification Loss: {epoch_loss / num_batches:.3f}')
    return epoch_loss / num_batches, epoch_acc / num_batches, epoch_results, \
           epoch_results_by_tag, epoch_CR


def evaluate(model, test_batch_generator, num_batches, device, label_map, class_weight):
    """
    Main evaluation routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_results = {}
    epoch_results_by_tag = {}
    epoch_CR = ""

    output_t_pred = None

    model.eval()
    with torch.no_grad():
        for b in tqdm(range(num_batches)):
            x_batch, y_batch, masks_batch = next(test_batch_generator)
            if len(x_batch.shape) == 3:
                x_batch = Variable(torch.FloatTensor(x_batch)).to(device)
            else:
                x_batch = Variable(torch.LongTensor(x_batch)).to(device)
            y_batch = y_batch.astype(float)
            y_batch = Variable(torch.LongTensor(y_batch)).to(device)
            masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
            class_weight = class_weight.to(device) if class_weight is not None else None
            outputs = model(input_ids=x_batch, attention_mask=masks_batch, labels=y_batch,
                            class_weight=class_weight)

            loss, logits = outputs[:2]

            if type(model) in [RobertaForTokenClassificationWithCRF, BiLSTMForTokenClassificationWithCRF]:
                y_batch = y_batch.detach().cpu()
                y_batch_filtered = [y_batch[i][y_batch[i] >= 0].tolist() for i in range(y_batch.shape[0])]
                eval_metrics = compute_crf_metrics(outputs[2], y_batch_filtered, label_map)
            else:
                eval_metrics = compute_metrics(logits.detach().cpu(), y_batch.detach().cpu(), label_map)

            epoch_loss += loss.item()
            epoch_acc += eval_metrics["accuracy_score"]
            epoch_results.update(eval_metrics["results"])
            epoch_results_by_tag.update(eval_metrics["results_by_tag"])
            epoch_CR = eval_metrics["CR"]
            if output_t_pred is None:
                output_t_pred = logits.detach().cpu().numpy()
            else:
                output_t_pred = np.concatenate([output_t_pred, logits.detach().cpu().numpy()], axis=0)

    return epoch_loss / num_batches, epoch_acc / num_batches, epoch_results, \
           epoch_results_by_tag, output_t_pred, epoch_CR


def predict(model, test_batch_generator, num_batches, device, label_map, class_weight):
    """
    Main evaluation routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_results = {}
    epoch_results_by_tag = {}
    epoch_CR = ""

    output_t_pred = None

    model.eval()
    with torch.no_grad():
        for b in tqdm(range(num_batches)):
            x_batch, y_batch, masks_batch = next(test_batch_generator)
            if len(x_batch.shape) == 3:
                x_batch = Variable(torch.FloatTensor(x_batch)).to(device)
            else:
                x_batch = Variable(torch.LongTensor(x_batch)).to(device)
            y_batch = y_batch.astype(float)
            y_batch = Variable(torch.LongTensor(y_batch)).to(device)
            masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
            class_weight = class_weight.to(device) if class_weight is not None else None

            outputs = model(input_ids=x_batch, attention_mask=masks_batch, labels=y_batch,
                            class_weight=class_weight)

            loss, logits = outputs[:2]

            if type(model) in [RobertaForTokenClassificationWithCRF, BiLSTMForTokenClassificationWithCRF]:
                y_batch = y_batch.detach().cpu()
                y_batch_filtered = [y_batch[i][y_batch[i] >= 0].tolist() for i in range(y_batch.shape[0])]
                eval_metrics = compute_crf_metrics(outputs[2], y_batch_filtered, label_map)
            else:
                eval_metrics = compute_metrics(logits.detach().cpu(), y_batch.detach().cpu(), label_map)

            epoch_loss += loss.item()
            epoch_acc += eval_metrics["accuracy_score"]
            epoch_results.update(eval_metrics["results"])
            epoch_results_by_tag.update(eval_metrics["results_by_tag"])
            epoch_CR = eval_metrics["CR"]
            if output_t_pred is None:
                output_t_pred = logits.detach().cpu().numpy()
            else:
                output_t_pred = np.concatenate([output_t_pred, logits.detach().cpu().numpy()], axis=0)

    return epoch_loss / num_batches, epoch_acc / num_batches, epoch_results, \
           epoch_results_by_tag, output_t_pred, epoch_CR


def load_model(model_type, model_path, config):
    if model_type == 'bertweet-token':
        model = RobertaForWeightedTokenClassification.from_pretrained(model_path, config=config)
    elif model_type == 'bertweet-token-crf':
        model = RobertaForTokenClassificationWithCRF.from_pretrained(model_path, config=config)
    elif model_type == 'BiLSTM-token':
        model = BiLSTMForWeightedTokenClassification(config=config)
        if model_path is not None:
            model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.pt')))
    elif model_type == 'BiLSTM-token-crf':
        model = BiLSTMForTokenClassificationWithCRF(config=config)
        if model_path is not None:
            model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.pt')))
    else:
        model = None
    return model


def get_embedding(text_list, embeddings_index, embeddings, max_length, token_label_raw_list, label_map):
    pad_token_label_id = -100
    output_embedding = []
    label_ids_list = []
    attention_masks_list = []
    for words, token_labels in zip(text_list, token_label_raw_list):
        words_mapped = [0] * max_length
        label_ids = [0] * max_length
        length = len(words)
        if (length < max_length):
            for i in range(0, length):
                words_mapped[i] = embeddings_index.get(words[i], -1)
                label_ids[i] = label_map[token_labels[i]]
            for i in range(length, max_length):
                words_mapped[i] = -2
                label_ids[i] = pad_token_label_id
        elif (length > max_length):
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
    parser.add_argument('--log_dir', default='log-BERTweet-token', type=str)
    parser.add_argument("--save_model", default=False, action='store_true')
    parser.add_argument("--early_stop", default=False, action='store_true')
    parser.add_argument("--freeze_bert", default=False, action='store_true')
    parser.add_argument("--miss_token_label", default=False, action='store_true')
    parser.add_argument("--extra_real_label", default=False, action='store_true')
    parser.add_argument("--fixed_epoch", default=False, action='store_true')
    parser.add_argument("--assign_weight", default=False, type='bool')
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
                    args.task_type + '/' + str(args.n_epochs) + '_epoch/' + \
                    args.data.split('/')[-1] + '/' + str(args.train_file).replace('/', '_') + '_' + \
                    str(args.val_file).replace('/', '_') + '_' + str(args.test_file).replace('/', '_') + '/' + \
                    str(args.assign_weight) + '_weight/' + str(args.freeze_bert) + '_freeze_bert/' + str(args.miss_token_label) + \
                    '_miss_token_label/' + str(args.extra_real_label) + '_extra_real_label/' + str(args.fixed_epoch) + \
                    '_fixed_epoch/' + str(args.seed) + '_seed/'
    log_filename = 'log.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.', '-') + '.txt'
    per_filename = 'performance.csv'
    model_dir = 'saved-model'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logname = os.path.join(log_directory, log_filename)
    modeldir = os.path.join(log_directory, model_dir)
    perfilename = os.path.join(log_directory, per_filename)
    wandb.init(
        # set the wandb project where this run will be logged
        project="Dissertation-Task-3",
        name=f"{args.model_type}-{args.bert_model}-{args.train_file}-{args.seed}",
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
    if args.miss_token_label is True:
        need_columns.append('att_label')
        if args.extra_real_label is True:
            need_columns.append('real_relevant_entity_label')
            X_train_raw, Y_train_raw, att_label_train_raw, real_token_label_train_raw = extract_from_dataframe(train_data, need_columns)
            X_dev_raw, Y_dev_raw, att_label_dev_raw, real_token_label_dev_raw = extract_from_dataframe(val_data, need_columns)
            X_test_raw, Y_test_raw = extract_from_dataframe(test_data, need_columns[:-2])
        else:
            X_train_raw, Y_train_raw, att_label_train_raw = extract_from_dataframe(train_data, need_columns)
            X_dev_raw, Y_dev_raw, att_label_dev_raw = extract_from_dataframe(val_data, need_columns)
            X_test_raw, Y_test_raw = extract_from_dataframe(test_data, need_columns[:-1])
    else:
        if args.extra_real_label is True:
            need_columns.append('real_relevant_entity_label')
            X_train_raw, Y_train_raw, real_token_label_train_raw = extract_from_dataframe(train_data, need_columns)
            X_dev_raw, Y_dev_raw, real_token_label_dev_raw = extract_from_dataframe(val_data, need_columns)
            X_test_raw, Y_test_raw = extract_from_dataframe(test_data, need_columns[:-1])
        else:
            X_train_raw, Y_train_raw = extract_from_dataframe(train_data, need_columns)
            X_dev_raw, Y_dev_raw = extract_from_dataframe(val_data, need_columns)
            X_test_raw, Y_test_raw = extract_from_dataframe(test_data, need_columns)
    args.eval_batch_size = X_dev_raw.shape[0]
    args.test_batch_size = X_test_raw.shape[0]

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
            X_train, masks_train, Y_train = tokenize_with_given_mask(
                X_train_raw, args.max_length, tokenizer, Y_train_raw, label_map, att_label_train_raw)
            X_dev, masks_dev, Y_dev = tokenize_with_given_mask(
                X_dev_raw, args.max_length, tokenizer, Y_dev_raw, label_map, att_label_dev_raw)
            X_test, masks_test, Y_test = tokenize_with_new_mask(
                X_test_raw, 128, tokenizer, Y_test_raw, label_map)
        else:
            X_train, masks_train, Y_train = tokenize_with_new_mask(
                X_train_raw, args.max_length, tokenizer, Y_train_raw, label_map)
            X_dev, masks_dev, Y_dev = tokenize_with_new_mask(
                X_dev_raw, args.max_length, tokenizer, Y_dev_raw, label_map)
            X_test, masks_test, Y_test = tokenize_with_new_mask(
                X_test_raw, 128, tokenizer, Y_test_raw, label_map)
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
            X_train, masks_train, Y_train = get_embedding_with_att(X_train_raw, embeddings_index, embeddings,
                                                                   args.max_length,
                                                                   Y_train_raw, label_map, att_label_train_raw)
            X_dev, masks_dev, Y_dev = get_embedding_with_att(X_dev_raw, embeddings_index, embeddings,
                                                             args.max_length,
                                                             Y_dev_raw, label_map, att_label_dev_raw)
            X_test, masks_test, Y_test = get_embedding(X_test_raw, embeddings_index, embeddings,
                                                       args.max_length,
                                                       Y_test_raw, label_map)
        else:
            X_train, masks_train, Y_train = get_embedding(X_train_raw, embeddings_index, embeddings,
                                                          args.max_length,
                                                          Y_train_raw, label_map)
            X_dev, masks_dev, Y_dev = get_embedding(X_dev_raw, embeddings_index, embeddings,
                                                    args.max_length,
                                                    Y_dev_raw, label_map)
            X_test, masks_test, Y_test = get_embedding(X_test_raw, embeddings_index, embeddings,
                                                       args.max_length,
                                                       Y_test_raw, label_map)
    # weight of each class in loss function
    class_weight = None
    if args.assign_weight:
        class_weight = [Y_train.shape[0] / (Y_train == i).sum() for i in range(len(labels))]
        class_weight = torch.FloatTensor(class_weight)

    if args.bert_model is not None:
        config = RobertaConfig.from_pretrained(args.bert_model)
        config.update({'num_labels': len(labels), })
    else:
        from dotmap import DotMap
        config = DotMap()
        config.update({'num_labels': len(labels),
                       'token_label_map': label_map, 'hidden_dropout_prob': 0.1,
                       'rnn_hidden_dimension': args.rnn_hidden_size})
    model = load_model(args.model_type, args.bert_model, config)

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

    best_valid_acc = 0
    best_valid_P, best_valid_R, best_valid_F = 0, 0, 0
    train_losses, eval_losses, r_train_losses, r_eval_losses = [], [], [], []
    train_F_list, eval_F_list, r_train_F_list, r_eval_F_list = [], [], [], []

    early_stop_sign = 0

    for epoch in range(args.n_epochs):

        # train

        train_batch_generator = mask_batch_generator(X_train, Y_train, masks_train, args.batch_size)
        num_batches = X_train.shape[0] // args.batch_size
        train_loss, train_acc, train_results, train_results_by_tag, train_CR = train(model, optimizer,
                                                                                     train_batch_generator,
                                                                                     num_batches,
                                                                                     device, args, label_map,
                                                                                     class_weight)
        train_batch_generator = mask_batch_seq_generator(X_train, Y_train, masks_train, args.batch_size)
        num_batches = X_train.shape[0] // args.batch_size
        train_loss, train_acc, train_results, train_results_by_tag, train_t_pred, train_CR = evaluate(model,
                                                                                                      train_batch_generator,
                                                                                                      num_batches,
                                                                                                      device,
                                                                                                      label_map,
                                                                                                      class_weight)
        train_losses.append(train_loss)
        train_F = train_results['strict']['f1']
        train_P = train_results['strict']['precision']
        train_R = train_results['strict']['recall']
        train_F_list.append(train_F)
        train_t_tuple = [train_loss, train_acc, train_results, train_results_by_tag, train_CR]
        train_t_dict = {'train_token_'+key: val for key, val in zip(['loss', 'acc', 'results', 'results_by_tag', 'CR'], train_t_tuple)}

        # eval
        dev_batch_generator = mask_batch_seq_generator(X_dev, Y_dev, masks_dev,
                                                       min(X_dev.shape[0], args.eval_batch_size))
        num_batches = X_dev.shape[0] // min(X_dev.shape[0], args.eval_batch_size)
        valid_loss, valid_acc, valid_results, valid_results_by_tag, valid_t_pred, valid_CR = evaluate(model,
                                                                                                      dev_batch_generator,
                                                                                                      num_batches,
                                                                                                      device,
                                                                                                      label_map,
                                                                                                      class_weight)
        eval_losses.append(valid_loss)
        valid_F = valid_results['strict']['f1']
        valid_P = valid_results['strict']['precision']
        valid_R = valid_results['strict']['recall']
        eval_F_list.append(valid_F)
        valid_t_tuple = [valid_loss, valid_acc, valid_results, valid_results_by_tag, valid_CR]
        valid_t_dict = {'valid_token_'+key: val for key, val in zip(['loss', 'acc', 'results', 'results_by_tag', 'CR'], valid_t_tuple)}

        r_train_t_dict, r_valid_t_dict = {}, {}
        # eval on real labels
        if args.extra_real_label is True:
            r_train_batch_generator = mask_batch_seq_generator(X_train, real_token_label_train, masks_train, args.batch_size)
            num_batches = X_train.shape[0] // args.batch_size
            r_train_loss, r_train_acc, r_train_results, r_train_results_by_tag, r_train_t_pred, r_train_CR = evaluate(model,
                                                                                                          r_train_batch_generator,
                                                                                                          num_batches,
                                                                                                          device,
                                                                                                          label_map,
                                                                                                          class_weight)
            r_train_losses.append(r_train_loss)
            r_train_F = r_train_results['strict']['f1']
            r_train_P = r_train_results['strict']['precision']
            r_train_R = r_train_results['strict']['recall']
            r_train_F_list.append(r_train_F)
            r_train_t_tuple = [r_train_loss, r_train_acc, r_train_results, r_train_results_by_tag]
            r_train_t_dict = {'real_train_token_'+key: val for key, val in zip(['loss', 'acc', 'results', 'results_by_tag', 'CR'], r_train_t_tuple)}
            

            r_dev_batch_generator = mask_batch_seq_generator(X_dev, real_token_label_dev, masks_dev,
                                                           min(X_dev.shape[0], args.eval_batch_size))
            num_batches = X_dev.shape[0] // min(X_dev.shape[0], args.eval_batch_size)
            r_valid_loss, r_valid_acc, r_valid_results, r_valid_results_by_tag, r_valid_t_pred, r_valid_CR = evaluate(model,
                                                                                                          r_dev_batch_generator,
                                                                                                          num_batches,
                                                                                                          device,
                                                                                                          label_map,
                                                                                                          class_weight)
            r_eval_losses.append(r_valid_loss)
            r_valid_F = r_valid_results['strict']['f1']
            r_valid_P = r_valid_results['strict']['precision']
            r_valid_R = r_valid_results['strict']['recall']
            r_eval_F_list.append(r_valid_F)
            r_valid_t_tuple = [r_valid_loss, r_valid_acc, r_valid_results, r_valid_results_by_tag, r_valid_CR]
            r_valid_t_dict = {'real_valid_token_'+key: val for key, val in zip(['loss', 'acc', 'results', 'results_by_tag', 'CR'], r_valid_t_tuple)}
        epoch_dict = {'epoch':epoch}
        for pdict in [train_t_dict, valid_t_dict, r_train_t_dict, r_valid_t_dict]:
            epoch_dict.update(pdict)
        wandb.log(epoch_dict)
        
        if (args.fixed_epoch and epoch == args.n_epochs - 1) or \
                ((not args.fixed_epoch) and (best_valid_F < valid_F) or epoch == 0):
            best_valid_acc = valid_acc
            best_valid_P = valid_P
            best_valid_R = valid_R
            best_valid_F = valid_F
            best_valid_results = valid_results
            best_valid_results_by_tag = valid_results_by_tag
            best_valid_CR = valid_CR

            best_train_acc = train_acc
            best_train_P = train_P
            best_train_R = train_R
            best_train_F = train_F
            best_train_results = train_results
            best_train_results_by_tag = train_results_by_tag
            best_train_CR = train_CR

            if args.extra_real_label is True:
                r_best_valid_acc = r_valid_acc
                r_best_valid_P = r_valid_P
                r_best_valid_R = r_valid_R
                r_best_valid_F = r_valid_F
                r_best_valid_results = r_valid_results
                r_best_valid_results_by_tag = r_valid_results_by_tag
                r_best_valid_CR = r_valid_CR

                r_best_train_acc = r_train_acc
                r_best_train_P = r_train_P
                r_best_train_R = r_train_R
                r_best_train_F = r_train_F
                r_best_train_results = r_train_results
                r_best_train_results_by_tag = r_train_results_by_tag
                r_best_train_CR = r_train_CR

            if args.bert_model is not None:
                model.save_pretrained(modeldir)
            else:
                torch.save(model.state_dict(), os.path.join(modeldir, 'pytorch_model.pt'))

            if (not args.fixed_epoch) and ((best_valid_F < valid_F) or (epoch == 0)):
                if args.early_stop:
                    early_stop_sign = 0
        elif (not args.fixed_epoch) and args.early_stop:
            early_stop_sign += 1

        print(f'Train Acc: {train_acc * 100:.2f}%')
        print(f'Train P: {train_P * 100:.2f}%')
        print(f'Train R: {train_R * 100:.2f}%')
        print(f'Train F1: {train_F * 100:.2f}%')
        print(f'Val. Acc: {valid_acc * 100:.2f}%')
        print(f'Val. P: {valid_P * 100:.2f}%')
        print(f'Val. R: {valid_R * 100:.2f}%')
        print(f'Val. F1: {valid_F * 100:.2f}%')
        logging.info(f'Train. Acc: {train_acc * 100:.2f}%')
        logging.info(f'Train. P: {train_P * 100:.2f}%')
        logging.info(f'Train. R: {train_R * 100:.2f}%')
        logging.info(f'Train. F1: {train_F * 100:.2f}%')
        logging.info(f'Val. Acc: {valid_acc * 100:.2f}%')
        logging.info(f'Val. P: {valid_P * 100:.2f}%')
        logging.info(f'Val. R: {valid_R * 100:.2f}%')
        logging.info(f'Val. F1: {valid_F * 100:.2f}%')

        if args.early_stop and early_stop_sign >= 5:
            break

    content = f"After {epoch + 1} epoch, Best valid F1: {best_valid_F}, accuracy: {best_valid_acc}, Recall: {best_valid_R}, Precision: {best_valid_P}"
    print(content)
    logging.info(content)

    performance_dict = vars(args)
    performance_dict['T_best_train_F'] = best_train_F
    performance_dict['T_best_train_ACC'] = best_train_acc
    performance_dict['T_best_train_R'] = best_train_R
    performance_dict['T_best_train_P'] = best_train_P
    performance_dict['T_best_train_CR'] = best_train_CR
    performance_dict['T_best_train_results'] = best_train_results
    performance_dict['T_best_train_results_by_tag'] = best_train_results_by_tag

    performance_dict['T_best_valid_F'] = best_valid_F
    performance_dict['T_best_valid_ACC'] = best_valid_acc
    performance_dict['T_best_valid_R'] = best_valid_R
    performance_dict['T_best_valid_P'] = best_valid_P
    performance_dict['T_best_valid_CR'] = best_valid_CR
    performance_dict['T_best_valid_results'] = best_valid_results
    performance_dict['T_best_valid_results_by_tag'] = best_valid_results_by_tag

    if args.extra_real_label is True:
        performance_dict['T_r_best_train_F'] = r_best_train_F
        performance_dict['T_r_best_train_ACC'] = r_best_train_acc
        performance_dict['T_r_best_train_R'] = r_best_train_R
        performance_dict['T_r_best_train_P'] = r_best_train_P
        performance_dict['T_r_best_train_CR'] = r_best_train_CR
        performance_dict['T_r_best_train_results'] = r_best_train_results
        performance_dict['T_r_best_train_results_by_tag'] = r_best_train_results_by_tag

        performance_dict['T_r_best_valid_F'] = r_best_valid_F
        performance_dict['T_r_best_valid_ACC'] = r_best_valid_acc
        performance_dict['T_r_best_valid_R'] = r_best_valid_R
        performance_dict['T_r_best_valid_P'] = r_best_valid_P
        performance_dict['T_r_best_valid_CR'] = r_best_valid_CR
        performance_dict['T_r_best_valid_results'] = r_best_valid_results
        performance_dict['T_r_best_valid_results_by_tag'] = r_best_valid_results_by_tag

    # Plot training classification loss
    epoch_count = np.arange(1, epoch + 2)
    fig, axs = plt.subplots(2, figsize=(10, 12), sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
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
    axs[1].plot(epoch_count, train_F_list, 'b--')
    axs[1].plot(epoch_count, eval_F_list, 'g-')
    axs[1].legend(['Training F1', 'Valid F1'], fontsize=14)
    if args.extra_real_label is True:
        axs[1].plot(epoch_count, r_train_F_list, 'r--')
        axs[1].plot(epoch_count, r_eval_F_list, 'm-')
        axs[1].legend(['Training F1', 'Valid F1', 'Real Training Loss', 'Real Valid Loss'], fontsize=14)
    axs[1].set_ylabel('F1', fontsize=16)
    axs[1].set_xlabel('Epoch', fontsize=16)
    axs[1].tick_params(axis='y', labelsize=14, labelcolor='r')
    axs[1].tick_params(axis='x', labelsize=14)

    figure_filename = 'fig.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.',
                                                                                                         '-') + '.png'
    figfullname = log_directory + figure_filename
    plt.savefig(figfullname, dpi=fig.dpi)

    num_batches = X_test.shape[0] // args.test_batch_size
    test_batch_generator = mask_batch_seq_generator(X_test, Y_test, masks_test, args.test_batch_size)
    del model
    torch.cuda.empty_cache()
    model = load_model(args.model_type, modeldir, config)
    model = model.to(device)
    test_loss, test_acc, test_results, test_results_by_tag, test_t_pred, test_CR = predict(model,
                                                                                           test_batch_generator,
                                                                                           num_batches, device,
                                                                                           label_map, class_weight)
    test_t_tuple = [test_loss, test_acc, test_results, test_results_by_tag, test_CR]
    test_t_dict = {'test_token_'+key: val for key, val in zip(['loss', 'acc', 'results', 'results_by_tag', 'CR'], test_t_tuple)}
    test_F = test_results['strict']['f1']
    test_P = test_results['strict']['precision']
    test_R = test_results['strict']['recall']
    print(f'Test Acc: {test_acc * 100:.2f}%')
    print(f'Test P: {test_P * 100:.2f}%')
    print(f'Test R: {test_R * 100:.2f}%')
    print(f'Test F1: {test_F * 100:.2f}%')
    logging.info(f'Test Acc: {test_acc * 100:.2f}%')
    logging.info(f'Test P: {test_P * 100:.2f}%')
    logging.info(f'Test R: {test_R * 100:.2f}%')
    logging.info(f'Test F1: {test_F * 100:.2f}%')
    token_pred_dir = log_directory + 'token_prediction.npy'
    if_crf = "crf" in args.model_type 
    if if_crf:
        with open(token_pred_dir, "wb") as fp:
            pickle.dump(test_t_pred, fp)
    else:
        np.save(token_pred_dir, test_t_pred)

    performance_dict['T_best_test_F'] = test_F
    performance_dict['T_best_test_ACC'] = test_acc
    performance_dict['T_best_test_R'] = test_R
    performance_dict['T_best_test_P'] = test_P
    performance_dict['T_best_test_CR'] = test_CR
    performance_dict['T_best_test_results'] = test_results
    performance_dict['T_best_test_results_by_tag'] = test_results_by_tag

    performance_dict['script_file'] = os.path.basename(__file__)
    performance_dict['log_directory'] = log_directory
    performance_dict['log_filename'] = log_filename
    performance_dict['note'] = NOTE
    performance_dict['Time'] = str(datetime.datetime.now())
    performance_dict['device'] = torch.cuda.get_device_name(device) if device.type != 'cpu' else 'cpu'
    wandb.log(test_t_dict)
    wandb.log({k:performance_dict[k] for k in ['script_file', 'log_directory', 'log_filename', 'note', 'Time', 'device']})
    for key, value in performance_dict.items():
        if type(value) is np.int64:
            performance_dict[key] = int(value)
    with open(args.performance_file, 'a+') as outfile:
        outfile.write(json.dumps(performance_dict) + '\n')
    if not args.save_model:
        shutil.rmtree(modeldir)


if __name__ == '__main__':
    main()
