from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model.crf import *
from model.lm_lstm_crf import *
import model.utils as utils
from model.predictor import predict_wc
import model.evaluator as Evaluator

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating LM-BLSTM-CRF')
    parser.add_argument('--load_arg', default='./checkpoint/cwlm_lstm_crf.json', help='path to arg json')
    parser.add_argument('--load_check_point', default='./checkpoint/cwlm_lstm_crf.model',
                        help='path to model checkpoint file')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id')
    parser.add_argument('--decode_type', choices=['label', 'string'], default='label',
                        help='type of decode function, set `label` to couple label with text, or set `string` to insert label into test')
    parser.add_argument('--batch_size', type=int, default=50, help='size of batch')
    parser.add_argument('--input_file', default='test.tsv', help='path to input un-annotated corpus')
    parser.add_argument('--output_file', default='annotate/output', help='path to output file')
    parser.add_argument('--dataset_no', type=int, default=5, help='number of the datasets')
    parser.add_argument('--scope', nargs='+', default=['ALL'],
                        help='the scope to predict. default: all tags in training')
    args = parser.parse_args()

    with open(args.load_arg, 'r') as f:
        jd = json.load(f)
    jd = jd['args']

    checkpoint_file = torch.load(args.load_check_point, map_location=lambda storage, loc: storage)
    f_map = checkpoint_file['f_map']
    l_map = checkpoint_file['l_map']
    c_map = checkpoint_file['c_map']
    in_doc_words = checkpoint_file['in_doc_words']
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    # build model
    ner_model = LM_LSTM_CRF(len(l_map), len(c_map), jd['char_dim'], jd['char_hidden'], jd['char_layers'],
                            jd['word_dim'], jd['word_hidden'], jd['word_layers'], len(f_map), jd['drop_out'],
                            args.dataset_no, large_CRF=jd['small_crf'], if_highway=jd['high_way'],
                            in_doc_words=in_doc_words, highway_layers=jd['highway_layers'])

    ner_model.load_state_dict(checkpoint_file['state_dict'])

    if args.gpu >= 0:
        if_cuda = True
        torch.cuda.set_device(args.gpu)
        ner_model.cuda()
        packer = CRFRepack_WC(len(l_map), True)
    else:
        if_cuda = False
        packer = CRFRepack_WC(len(l_map), False)

    decode_label = (args.decode_type == 'label')
    predictor = predict_wc(if_cuda, f_map, c_map, l_map, f_map['<eof>'], c_map['\n'], l_map['<pad>'], l_map['<start>'],
                           decode_label, args.batch_size, jd['caseless'])

    # loading corpus
    lines = []
    features = []
    labels = []
    with codecs.open(args.input_file, 'r', 'utf-8') as f:
        for line in f:
            if line == '\n':
                f, l = utils.read_corpus(lines)
                features.append([f])
                labels.append(l)
                lines = []
                continue
            lines.append(line)

    # for idx in range(args.dataset_no):
    #     print('annotating the entity type', idx)
    #     fout = open(args.output_file+str(idx)+'.txt', 'w')
    #     for feature in features:
    #         predictor.output_batch(ner_model, feature, fout, idx)
    #         # predictor.combined_output_batch(ner_model, feature, fout)
    #         fout.write('\n')
    #     fout.close()

    evaluator = Evaluator.eval_sentence()
    fout = open(args.output_file + str('_combine') + '.txt', 'w')
    for idx in range(len(features)):
        feature = features[idx]
        label = labels[idx]
        combined_pred = predictor.combined_output_batch(ner_model, feature, fout, 5, args.scope)
        # replace = [('-cell_type', '-celltype'),
        #            ('-GENE', '-geneprotein'),
        #            ('-cell_line', '-cellline'),
        #            ('-protein', '-geneprotein')]
        replace = []
        combined_pred = utils.unify(combined_pred, replace)

        evaluator.eval_sent(combined_pred, label[0])
    fout.close()
    f = open('output', 'a')
    f.write(args.input_file)
    f.write('\n')
    F1, prec, recall, acc = evaluator.f1_score()
    f.write("{0} {1} {2} {3}".format(F1, prec, recall, acc))
    f.write('\n')
    f.write('\n')
