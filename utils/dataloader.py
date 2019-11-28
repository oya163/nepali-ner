#!/usr/bin/env python3

'''
    NER Dataloader
    Author: Oyesh Mann Singh
    Date: 10/14/2019

    Data format:
        <WORD>  <NER-tag>

'''

import io
import os
import logging
import numpy as np

import torch
import torchtext
from torchtext import data
from torchtext import vocab
from torchtext import datasets
from torchtext.datasets import SequenceTaggingDataset

from uniseg.graphemecluster import grapheme_clusters

class Dataloader():
    def __init__(self, config, k):
        self.root_path = os.path.join(config.root_path, k)
        self.batch_size = config.batch_size
        self.device = config.device
        self.use_pos = config.use_pos
        
        self.txt_field = data.Field(tokenize=list, use_vocab=True, unk_token='<unk>', batch_first=True)
        self.label_field = data.Field(unk_token=None, batch_first=True)
        self.char_field = data.Field(unk_token='<unk>', sequential=False)
        self.graph_field = data.Field(unk_token='<unk>', sequential=False)
        
        self.fields = (('TEXT', self.txt_field), ('LABEL', self.label_field))
        
        if config.use_pos:
            self.pos_field = data.Field(unk_token=None, batch_first=True)
            self.fields = (('TEXT', self.txt_field), ('POS', self.pos_field), ('LABEL', self.label_field))

        
        self.train_ds, self.val_ds, self.test_ds = SequenceTaggingDataset.splits(path=self.root_path,
                                                    fields=self.fields, separator='\t',
                                                    train='train.txt', validation='val.txt',
                                                    test='test.txt')

        self.char_list = []
        self.graph_list = []
        for each in self.train_ds.examples + self.test_ds.examples + self.val_ds.examples:
            for x in each.TEXT:
                self.char_list+=list(x)
                self.graph_list+=list(grapheme_clusters(x))
        self.char_list = list(set(self.char_list))
        self.graph_list = list(set(self.graph_list))

        self.graph_list.sort()
        self.char_list.sort()
        
        self.embedding_dir = config.emb_dir
        self.vec = vocab.Vectors(name=config.emb_file, cache=self.embedding_dir)

        self.txt_field.build_vocab(self.train_ds, self.test_ds, self.val_ds, max_size=None, vectors=self.vec)
        self.label_field.build_vocab(self.train_ds.LABEL, self.test_ds.LABEL, self.val_ds.LABEL)
        
        if config.char_pretrained:
            self.char_vec = vocab.Vectors(name=config.char_emb_file, cache=self.embedding_dir)
            self.graph_vec = vocab.Vectors(name=config.graph_emb_file, cache=self.embedding_dir)
            
            self.char_field.build_vocab(self.char_list, vectors=self.char_vec)
            self.graph_field.build_vocab(self.graph_list, vectors=self.graph_vec)
        else:
            self.char_field.build_vocab(self.char_list)
            self.graph_field.build_vocab(self.graph_list)
        
        
        self.vocab_size = len(self.txt_field.vocab)
        self.tagset_size = len(self.label_field.vocab)
        self.char_vocab_size = len(self.char_field.vocab)
        self.graph_vocab_size = len(self.graph_field.vocab)
        
        self.weights = self.txt_field.vocab.vectors
        self.char_weights = self.char_field.vocab.vectors
        self.graph_weights = self.graph_field.vocab.vectors
        
        if config.use_pos:
            self.pos_field.build_vocab(self.train_ds.POS, self.test_ds.POS, self.val_ds.POS)
            # Because len(pos) = 56 and len(pos_field.vocab) = 55
            self.pos_size = len(self.pos_field.vocab) + 2
            self.pos_one_hot = np.eye(self.pos_size)
            self.one_hot_weight = torch.from_numpy(self.pos_one_hot).float()
        
        if config.verbose:
            self.print_stat()

        
    def train_ds(self):
        return self.train_ds
    
    def val_ds(self):
        return self.val_ds    
    
    def test_ds(self):
        return self.test_ds
    
    def txt_field(self):
        return self.txt_field
    
    def label_field(self):
        return self.label_field    
    
    def vocab_size(self):
        return self.vocab_size

    def tagset_size(self):
        return self.tagset_size

    def pos_size(self):
        return self.pos_size

    def weights(self):
        return self.weights

    def char_weights(self):
        return self.char_weights
    
    def graph_weights(self):
        return self.graph_weights    
    
    def get_char_vocab_size(self):
        return self.char_vocab_size
    
    def get_chars(self):
        return self.char_list

    def get_graph_vocab_size(self):
        return self.graph_vocab_size
    
    def get_graph(self):
        return self.graph_list
    
    def print_stat(self):
        print('Length of text vocab (unique words in dataset) = ', self.vocab_size)
        print('Length of label vocab (unique tags in labels) = ', self.tagset_size)
        if self.use_pos:
            print('Length of POS vocab (unique tags in POS) = ', self.pos_size)
        print('Length of char vocab (unique characters in dataset) = ', self.char_vocab_size)
        print('Length of grapheme vocab (unique graphemes in dataset) = ', self.graph_vocab_size)
    
    def load_data(self, batch_size, shuffle=True):
        train_iter, val_iter, test_iter = data.BucketIterator.splits(datasets=(self.train_ds, self.val_ds, self.test_ds), 
                                            batch_sizes=(batch_size, batch_size, batch_size), 
                                            sort_key=lambda x: len(x.TEXT), 
                                            device=self.device, 
                                            sort_within_batch=True, 
                                            repeat=False,
                                            shuffle=True)

        return train_iter, val_iter, test_iter

        

        
