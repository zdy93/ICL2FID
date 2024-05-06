from utils import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import argparse
import os, os.path
import pickle

from transformers import AutoTokenizer
from compute_multi_performance import print_table, calculate_mean_ci, new_compute_metrics


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


def label_tokenize_with_new_mask(orig_text, max_length, tokenizer, orig_labels, label_map):
    """
    tokenize a array of raw text and generate corresponding
    attention labels array and attention masks array
    """
    pad_token_label_id = -100
    simple_tokenize_results = [list(tt) for tt in zip(
        *[simple_tokenize(orig_text[i], tokenizer, orig_labels[i], label_map, max_length) for i in
          range(len(orig_text))])]
    bert_tokens, label_ids = simple_tokenize_results[0], simple_tokenize_results[1]
    label_ids = pad_sequences(label_ids, maxlen=max_length, dtype="long", truncating="post", padding="post",
                              value=pad_token_label_id)

    return label_ids


def get_label(text_list, max_length, token_label_raw_list, label_map):
    pad_token_label_id = -100
    label_ids_list = []
    for words, token_labels in zip(text_list, token_label_raw_list):
        label_ids = [0] * max_length
        length = len(words)
        if length < max_length:
            for i in range(0, length):
                label_ids[i] = label_map[token_labels[i]]
            for i in range(length, max_length):
                label_ids[i] = pad_token_label_id
        elif length > max_length:
            print('We should never see this print either')
        else:
            for i in range(0, max_length):
                label_ids[i] = label_map[token_labels[i]]

        label_ids_list.append(label_ids)
    label_ids_list = np.array(label_ids_list)
    return label_ids_list


def get_bootstrap_report(test_t_label, test_t_pred, if_crf, label_map, bootstrap_time):
    rng = np.random.RandomState(seed=12345)
    idx = np.arange(len(test_t_label))
    if if_crf:
        test_t_pred, test_t_label = align_predictions_crf(test_t_pred, test_t_label, label_map)
    else:
        test_t_pred, test_t_label = align_predictions(test_t_pred, test_t_label, label_map)
    t_strict_f1_list, t_default_f1_list, t_strict_bacc_list = [], [], []
    for i in tqdm(range(bootstrap_time)):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        test_t_pred_keep = [test_t_pred[i] for i in pred_idx]
        test_t_label_keep = [test_t_label[i] for i in pred_idx]
        t_eval_metrics = new_compute_metrics(test_t_pred_keep, test_t_label_keep)
        t_strict_f1_list.append(t_eval_metrics["strict_f1"])
        t_default_f1_list.append(t_eval_metrics["default_f1"])
        t_strict_bacc_list.append(t_eval_metrics["strict_bacc"])
    t_dict = {"strict_f1": calculate_mean_ci(t_strict_f1_list),
              "default_f1": calculate_mean_ci(t_default_f1_list),
              "strict_bacc": calculate_mean_ci(t_strict_bacc_list)}

    return t_dict


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
    parser.add_argument("--do_bootstrap", type='bool', help="do bootstrap on test set to get CI")
    parser.add_argument("--bootstrap_time", type=int, default=200, help="number of bootstrap")

    args = parser.parse_args()

    assert args.task_type in ['entity_detection', 'relevant_entity_detection', 'entity_relevance_classification']

    log_directory = args.log_dir + '/' + str(args.bert_model).split('/')[-1] + '/' + args.model_type + '/' + \
                    args.task_type + '/' + str(args.n_epochs) + '_epoch/' + \
                    args.data.split('/')[-1] + '/' + str(args.train_file).replace('/', '_') + '_' + \
                    str(args.val_file).replace('/', '_') + '_' + str(args.test_file).replace('/', '_') + '/' + \
                    str(args.assign_weight) + '_weight/' + str(args.freeze_bert) + '_freeze_bert/' + str(
        args.miss_token_label) + \
                    '_miss_token_label/' + str(args.extra_real_label) + '_extra_real_label/' + str(args.fixed_epoch) + \
                    '_fixed_epoch/' + str(args.seed) + '_seed/'

    print(log_directory)
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
            X_test_raw, Y_test_raw = extract_from_dataframe(test_data, need_columns[:-2])
        else:
            X_test_raw, Y_test_raw = extract_from_dataframe(test_data, need_columns[:-1])
    else:
        if args.extra_real_label is True:
            need_columns.append('real_relevant_entity_label')
            X_test_raw, Y_test_raw = extract_from_dataframe(test_data, need_columns[:-1])
        else:
            X_test_raw, Y_test_raw = extract_from_dataframe(test_data, need_columns)

    with open(os.path.join(args.data, args.label_map), 'r') as fp:
        label_map = json.load(fp)

    labels = list(label_map.keys())

    if args.bert_model is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model, normalization=True)

        if args.miss_token_label is True:
            Y_test = label_tokenize_with_new_mask(
                X_test_raw, 128, tokenizer, Y_test_raw, label_map)
        else:
            Y_test = label_tokenize_with_new_mask(
                X_test_raw, 128, tokenizer, Y_test_raw, label_map)
    else:
        if args.miss_token_label is True:
            Y_test = get_label(X_test_raw, args.max_length,
                                                   Y_test_raw, label_map)
        else:
            Y_test = get_label(X_test_raw, args.max_length,
                                                   Y_test_raw, label_map)
    if_crf = "crf" in args.model_type
    token_pred_dir = log_directory + 'token_prediction.npy'
    if if_crf:
        with open(token_pred_dir, "rb") as fp:
            test_t_pred = pickle.load(fp)
    else:
        test_t_pred = np.load(token_pred_dir)

    if if_crf:
        token_label_test = [Y_test[i][Y_test[i] >= 0].tolist() for i in
                            range(Y_test.shape[0])]
        t_eval_metrics = compute_crf_metrics(test_t_pred, token_label_test, label_map)
    else:
        t_eval_metrics = compute_metrics(test_t_pred, Y_test, label_map)

    t_f1, t_pre, t_rec = [t_eval_metrics["results"]["strict"][key] for key in ["f1", "precision", "recall"]]

    data_dict = {"Task": "Word Level", "model_type": args.model_type, "assign_weight": args.assign_weight,
                 "data": args.data, "bert_model": args.bert_model, "train_file": args.train_file, "test_file": args.test_file,
                 "F1": t_f1, "precision": t_pre, "recall": t_rec}
    print_table(data_dict)

    if args.do_bootstrap:
        t_table = get_bootstrap_report(Y_test, test_t_pred, if_crf, label_map, args.bootstrap_time)

        t_table.update({"Task": "Word Level", "model_type": args.model_type, "assign_weight": args.assign_weight,
                        "data": args.data, "bert_model": args.bert_model, "train_file": args.train_file, "test_file": args.test_file})
        print_table(t_table)


if __name__ == '__main__':
    main()
