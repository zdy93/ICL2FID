from utils import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os, os.path

from compute_multi_performance import print_table, calculate_mean_ci, eval_metrics


def get_bootstrap_report(test_s_label, test_s_pred, bootstrap_time):
    rng = np.random.RandomState(seed=12345)
    idx = np.arange(test_s_label.shape[0])
    s_acc_list, s_bacc_list, s_f1_list, s_pre_list, s_rec_list = [], [], [], [], []
    for i in tqdm(range(bootstrap_time)):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        s_acc, s_bacc, s_f1, s_pre, s_rec = eval_metrics(test_s_pred[pred_idx], test_s_label[pred_idx])
        s_acc_list.append(s_acc)
        s_bacc_list.append(s_bacc)
        s_f1_list.append(s_f1)
        s_pre_list.append(s_pre)
        s_rec_list.append(s_rec)
    s_dict = {"accuracy": calculate_mean_ci(s_acc_list),
              "balanced_accuracy": calculate_mean_ci(s_bacc_list),
              "f1": calculate_mean_ci(s_f1_list),
              "precision": calculate_mean_ci(s_pre_list),
              "recall": calculate_mean_ci(s_rec_list)}

    return s_dict


def main():
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument("--bert_model", default=None, type=str)
    parser.add_argument("--model_type", default=None, type=str)
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--rnn_hidden_size', default=384, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=300, type=int)
    parser.add_argument('--test_batch_size', default=300, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--data', default='wnut_16', type=str)
    parser.add_argument('--log_dir', default='log-BERTweet-seq', type=str)
    parser.add_argument("--save_model", default=False, action='store_true')
    parser.add_argument("--early_stop", default=False, action='store_true')
    parser.add_argument("--freeze_bert", default=False, action='store_true')
    parser.add_argument("--fixed_epoch", default=False, action='store_true')
    parser.add_argument("--assign_weight", default=False, type='bool')
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--val_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--performance_file", default='all_test_performance.txt', type=str)
    parser.add_argument("--embeddings_file", default='glove.840B.300d.txt', type=str)
    parser.add_argument("--do_bootstrap", type='bool', help="do bootstrap on test set to get CI")
    parser.add_argument("--bootstrap_time", type=int, default=200, help="number of bootstrap")

    args = parser.parse_args()

    log_directory = args.log_dir + '/' + str(args.bert_model).split('/')[-1] + '/' + args.model_type + '/' \
                    + str(args.n_epochs) + '_epoch/' + args.data.split('/')[-1] + '/' + \
                    '/' + str(args.train_file).replace('/', '_') + '_' + str(args.val_file).replace('/', '_') + \
                    '_' + str(args.test_file).replace('/', '_') + '/' + \
                    str(args.assign_weight) + '_weight/' + str(args.freeze_bert) + \
                    '_freeze_bert/' + str(args.fixed_epoch) + '_fixed_epoch/' + str(args.seed) + '_seed/'

    print(log_directory)
    test_data = pd.read_pickle(os.path.join(args.data, args.test_file))
    need_columns = ['tweet_tokens', 'sentence_class']
    X_test_raw, Y_test = extract_from_dataframe(test_data, need_columns)
    seq_pred_dir = log_directory + 'seq_prediction.npy'
    test_s_pred = np.load(seq_pred_dir)

    s_acc, s_bacc, s_f1, s_pre, s_rec = eval_metrics(test_s_pred, Y_test)

    data_dict = {"Task": "Tweet Level", "model_type": args.model_type, "assign_weight": args.assign_weight,
                 "data": args.data, "bert_model": args.bert_model, "train_file": args.train_file,
                 "test_file": args.test_file,
                 "balanced_accuracy": s_bacc, "accuracy": s_acc, "F1": s_f1, "precision": s_pre, "recall": s_rec}
    print_table(data_dict)

    if args.do_bootstrap:
        s_table = get_bootstrap_report(Y_test, test_s_pred, args.bootstrap_time)
        s_table.update({"Task": "Tweet Level", "model_type": args.model_type, "assign_weight": args.assign_weight,
                        "data": args.data, "bert_model": args.bert_model, "train_file": args.train_file,
                        "test_file": args.test_file})
        print_table(s_table)


if __name__ == '__main__':
    main()
