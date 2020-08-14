#!/usr/bin/env python3

'''
    NER Dataloader
    Author: Oyesh Mann Singh
    Date: 10/14/2019

    Data format:
        <WORD>  <NER-tag>

'''

import os
import pickle
from torchtext import data, vocab
from torchtext.datasets import SequenceTaggingDataset


class Dataloader():
    def __init__(self, config, k):
        self.device = config.device
        self.use_pos = config.use_pos

        self.txt_field = pickle.load(open(config.vocab_file, 'rb'))
        self.label_field = pickle.load(open(config.label_file, 'rb'))

        # Save vocab and label file like this
        # For future reference
        # output = open('vocab.pkl', 'wb')
        # pickle.dump(self.txt_field, output)

        # output_label = open('labels.pkl', 'wb')
        # pickle.dump(self.label_field, output_label)

        self.vocab_size = len(self.txt_field.vocab)
        self.tagset_size = len(self.label_field.vocab)

        self.weights = self.txt_field.vocab.vectors


    def tokenizer(self, x):
        return x.split()

    # def train_ds(self):
    #     return self.train_ds
    #
    # def val_ds(self):
    #     return self.val_ds
    #
    # def test_ds(self):
    #     return self.test_ds

    def txt_field(self):
        return self.txt_field

    def label_field(self):
        return self.label_field

    def vocab_size(self):
        return self.vocab_size

    def tagset_size(self):
        return self.tagset_size

    def weights(self):
        return self.weights

    def print_stat(self):
        print('Length of text vocab (unique words in dataset) = ', self.vocab_size)
        print('Length of label vocab (unique tags in labels) = ', self.tagset_size)
        # if self.use_pos:
        #     print('Length of POS vocab (unique tags in POS) = ', self.pos_size)
        # print('Length of char vocab (unique characters in dataset) = ', self.char_vocab_size)
        # print('Length of grapheme vocab (unique graphemes in dataset) = ', self.graph_vocab_size)

    # def load_data(self, batch_size, shuffle=True):
    #     train_iter, val_iter, test_iter = data.BucketIterator.splits(
    #         datasets=(self.train_ds, self.val_ds, self.test_ds),
    #         batch_sizes=(batch_size, batch_size, batch_size),
    #         sort_key=lambda x: len(x.TEXT),
    #         device=self.device,
    #         sort_within_batch=True,
    #         repeat=False,
    #         shuffle=True)
    #
    #     return train_iter, val_iter, test_iter
