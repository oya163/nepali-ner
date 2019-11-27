'''
    Writes result into the file
    Author: Oyesh Mann Singh
'''

import os
import logging
import numpy as np

import torchtext
from torchtext import data
from torchtext import vocab

import torch
import torch.nn as nn

from tqdm import tqdm, tqdm_notebook, tnrange
tqdm.pandas(desc='Progress')

import utils.conlleval_perl as e

class Evaluator():
    def __init__(self, config, logger, model, dataloader, model_name):
        self.config = config
        self.logger = logger
        self.model = model
        self.model_name = model_name
        self.dataloader = dataloader
        self.use_pos = config.use_pos
        
        self.train_dl, self.val_dl, self.test_dl = dataloader.load_data(batch_size=1, shuffle=False)
        self.results_dir = config.results_dir
        
        tr_file = self.model_name+'_train.txt'
        ts_file = self.model_name+'_test.txt'
        vl_file = self.model_name+'_val.txt'
        
        self.train_file = os.path.join(self.results_dir, tr_file)
        self.test_file = os.path.join(self.results_dir, ts_file)
        self.val_file = os.path.join(self.results_dir, vl_file)
        
    def numpy_to_sent(self, tensor):
        '''
            Returns the corresponding TEXT of given Predictions
            Returns chunks of string
        '''    
        return ' '.join([self.dataloader.txt_field.vocab.itos[i] for i in tensor.cpu().data.numpy()[0]]).split()


    def pred_to_tag(self, predictions):
        '''
            Returns the corresponding TAGS of given Predictions
            Returns chunks of string
        '''
        return ' '.join([self.dataloader.label_field.vocab.itos[i] for i in predictions]).split()         
        
        
    def write_results(self):
        with open(self.train_file, 'w', encoding='utf-8') as rtrn:
            self.logger.info('Writing in file: {0}'.format(self.train_file))
            tr = tqdm(iter(self.train_dl), leave=False)
            for (k, v) in tr:
                if self.use_pos:
                    (X, p, y) = k
                    pred = self.model(X, p)
                else:
                    (X, y) = k
                    pred = self.model(X, None)
                sent = self.numpy_to_sent(X)
                pred_idx = torch.max(pred, 1)[1]

                y = y.view(-1)
                y_true_val = y.cpu().data.numpy().tolist()
                true_tag = self.pred_to_tag(y_true_val)

                y_pred_val = pred_idx.cpu().data.numpy().tolist()
                pred_tag = self.pred_to_tag(y_pred_val)

                for s, gt, pt in zip(sent, true_tag, pred_tag):
                    rtrn.write(s+' '+gt+' '+pt+'\n')
                rtrn.write('\n')
        rtrn.close()

        with open(self.test_file, 'w', encoding='utf-8') as rtst:
            self.logger.info('Writing in file: {0}'.format(self.test_file))
            tt = tqdm(iter(self.test_dl), leave=False)
            for (k, v) in tt:
                if self.use_pos:
                    (X, p, y) = k
                    pred = self.model(X, p)
                else:
                    (X, y) = k
                    pred = self.model(X, None)
                sent = self.numpy_to_sent(X)
                
                pred_idx = torch.max(pred, 1)[1]

                y = y.view(-1)
                y_true_val = y.cpu().data.numpy().tolist()
                true_tag = self.pred_to_tag(y_true_val)

                y_pred_val = pred_idx.cpu().data.numpy().tolist()
                pred_tag = self.pred_to_tag(y_pred_val)

                for s, gt, pt in zip(sent, true_tag, pred_tag):
                    rtst.write(s+' '+gt+' '+pt+'\n')
                rtst.write('\n')
        rtst.close()

        with open(self.val_file, 'w', encoding='utf-8') as rval:
            self.logger.info('Writing in file: {0}'.format(self.val_file))
            vl = tqdm(iter(self.val_dl), leave=False)
            for (k, v) in vl:
                if self.use_pos:
                    (X, p, y) = k
                    pred = self.model(X, p)
                else:
                    (X, y) = k
                    pred = self.model(X, None)
                sent = self.numpy_to_sent(X)
                
                pred_idx = torch.max(pred, 1)[1]

                y = y.view(-1)
                y_true_val = y.cpu().data.numpy().tolist()
                true_tag = self.pred_to_tag(y_true_val)

                y_pred_val = pred_idx.cpu().data.numpy().tolist()
                pred_tag = self.pred_to_tag(y_pred_val)

                for s, gt, pt in zip(sent, true_tag, pred_tag):
                    rval.write(s+' '+gt+' '+pt+'\n')
                rval.write('\n')
        rval.close()  
        
        
    def conll_eval(self):
        """
            Prints CoNLL Evaluation Report
        """
        acc, prec, rec, f1 = e.evaluate_conll_file(logger = self.logger,
                                                   fileName = self.test_file,
                                                   raw = True,
                                                   delimiter = None,
                                                   oTag = 'O',
                                                   latex = False)
        return (acc, prec, rec, f1)
