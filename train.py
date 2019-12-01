#!/usr/bin/env python3

'''
    Trainer
    Author: Oyesh Mann Singh
'''

import os
import argparse
import logging
import pandas as pd
import numpy as np
from utils.dataloader import Dataloader
from utils.eval import Evaluator

from tqdm import tqdm, tqdm_notebook, tnrange
tqdm.pandas(desc='Progress')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
torch.manual_seed(163)

from sklearn.metrics import accuracy_score

# Decay functions to be used with lr_scheduler
def lr_decay_noam(config):
    return lambda t: (
        10.0 * config.hidden_dim**-0.5 * min(
        (t + 1) * config.learning_rate_warmup_steps**-1.5, (t + 1)**-0.5))

def lr_decay_exp(config):
    return lambda t: config.learning_rate_falloff ** t


# Map names to lr decay functions
lr_decay_map = {
    'noam': lr_decay_noam,
    'exp': lr_decay_exp
}

class Trainer():
    def __init__(self, config, logger, dataloader, model, k):
        self.config = config
        self.logger = logger
        self.dataloader = dataloader
        self.verbose = config.verbose
        self.use_pos = config.use_pos
        
        self.train_dl, self.val_dl, self.test_dl = dataloader.load_data(batch_size=config.batch_size)

        ### DO NOT DELETE
        ### DEBUGGING PURPOSE
#         sample = next(iter(self.train_dl))
#         print(sample.TEXT)
#         print(sample.LABEL)
#         print(sample.POS)
        
        self.train_dlen = len(self.train_dl)
        self.val_dlen = len(self.val_dl)
        self.test_dlen = len(self.test_dl)
        
        self.model = model
        self.epochs = config.epochs
        
        self.loss_fn = nn.NLLLoss()

        self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                         lr=config.learning_rate, 
                         weight_decay=config.weight_decay)
        
        self.lr_scheduler_step = self.lr_scheduler_epoch = None
        
        # Set up learing rate decay scheme
        if config.use_lr_decay:
            if '_' not in config.lr_rate_decay:
                raise ValueError("Malformed learning_rate_decay")
            lrd_scheme, lrd_range = config.lr_rate_decay.split('_')

            if lrd_scheme not in lr_decay_map:
                raise ValueError("Unknown lr decay scheme {}".format(lrd_scheme))
            
            lrd_func = lr_decay_map[lrd_scheme]            
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                                            self.opt, 
                                            lrd_func(config),
                                            last_epoch=-1
                                        )
            # For each scheme, decay can happen every step or every epoch
            if lrd_range == 'epoch':
                self.lr_scheduler_epoch = lr_scheduler
            elif lrd_range == 'step':
                self.lr_scheduler_step = lr_scheduler
            else:
                raise ValueError("Unknown lr decay range {}".format(lrd_range))
                
                
        

    
        self.k = k
        self.model_name=config.model_name + self.k
        self.file_name = self.model_name + '.pth'
        self.model_file = os.path.join(config.output_dir, self.file_name)
        
        self.total_train_loss = []
        self.total_train_acc = []
        self.total_val_loss = []
        self.total_val_acc = []
        
        self.early_max_patience = config.early_max_patience
        
        
    def load_checkpoint(self):
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.opt = checkpoint['opt']
        self.opt.load_state_dict(checkpoint['opt_state'])
        self.total_train_loss = checkpoint['train_loss']
        self.total_train_acc = checkpoint['train_acc']
        self.total_val_loss = checkpoint['val_loss']
        self.total_val_acc = checkpoint['val_acc']
        self.epochs = checkpoint['epochs']
        
        
    def save_checkpoint(self):
        save_parameters = {'state_dict': self.model.state_dict(),
                           'opt': self.opt,
                           'opt_state': self.opt.state_dict(),
                           'train_loss' : self.total_train_loss,
                           'train_acc' : self.total_train_acc,
                           'val_loss' : self.total_val_loss,
                           'val_acc' : self.total_val_acc,
                           'epochs' : self.epochs}
        torch.save(save_parameters, self.model_file)        


    def fit(self):
        prev_lstm_val_acc = 0.0
        prev_val_loss = 100.0
        counter = 0
        patience_limit = 10
        
        for epoch in tnrange(0, self.epochs):      
            y_true_train = list()
            y_pred_train = list()
            total_loss_train = 0          

            t = tqdm(iter(self.train_dl), leave=False, total=self.train_dlen)
            for (k, v) in t:
                t.set_description(f'Epoch {epoch+1}')     
                self.model.train()
                
                self.opt.zero_grad()
                
                if self.use_pos:
                    (X, p, y) = k
                    pred = self.model(X, p)
                else:
                    (X, y) = k
                    pred = self.model(X, None)
                    
                y = y.view(-1)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.opt.step()

                if self.lr_scheduler_step:
                    self.lr_scheduler_step.step()
                
                t.set_postfix(loss=loss.item())
                pred_idx = torch.max(pred, dim=1)[1]

                y_true_train += list(y.cpu().data.numpy())
                y_pred_train += list(pred_idx.cpu().data.numpy())
                total_loss_train += loss.item()
            
                
            train_acc = accuracy_score(y_true_train, y_pred_train)
            train_loss = total_loss_train/self.train_dlen
            self.total_train_loss.append(train_loss)
            self.total_train_acc.append(train_acc)

            if self.val_dl:
                y_true_val = list()
                y_pred_val = list()
                total_loss_val = 0
                v = tqdm(iter(self.val_dl), leave=False)
                for (k, v) in v:
                    if self.use_pos:
                        (X, p, y) = k
                        pred = self.model(X, p)
                    else:
                        (X, y) = k
                        pred = self.model(X, None)
                    y = y.view(-1)
                    loss = self.loss_fn(pred, y)
                    pred_idx = torch.max(pred, 1)[1]
                    y_true_val += list(y.cpu().data.numpy())
                    y_pred_val += list(pred_idx.cpu().data.numpy())
                    total_loss_val += loss.item()

                valacc = accuracy_score(y_true_val, y_pred_val)
                valloss = total_loss_val/self.val_dlen
                self.logger.info(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {valloss:.4f} val_acc: {valacc:.4f}')
            else:
                self.logger.info(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f}')
            self.total_val_loss.append(valloss)
            self.total_val_acc.append(valacc)

            if self.lr_scheduler_epoch:
                self.lr_scheduler_epoch.step()
                
            if valloss < prev_val_loss:
                self.save_checkpoint()
                prev_val_loss = valloss
                counter=0
                self.logger.info("Best model saved!!!")
            else: 
                counter += 1

            if counter >= self.early_max_patience: 
                self.logger.info("Training stopped because maximum tolerance reached!!!")
                break

    
    
    # Predict
    def predict(self):
        self.model.eval()
        evaluate = Evaluator(self.config, self.logger, self.model, self.dataloader, self.model_name)
        self.logger.info("Writing results")
        evaluate.write_results()
        self.logger.info("Evaluate results")
        acc, prec, rec, f1 = evaluate.conll_eval()
        return (acc, prec, rec, f1)
    
