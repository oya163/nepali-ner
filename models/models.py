'''
    Models
    Author: Oyesh Mann Singh
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from uniseg.graphemecluster import grapheme_clusters

tqdm.pandas(desc='Progress')


class LSTMInferer(nn.Module):
    def __init__(self, config):
        super(LSTMInferer, self).__init__()
        self.bidirectional = config.bidirection
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        self.hidden_dim = config.hidden_dim
        self.vocab_size = 11764
        self.tagset_size = 8
        self.embedding_dim = config.embedding_dim
        self.device = config.device
        self.use_pos = config.use_pos

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        if self.use_pos:
            self.pos_size = 8
            self.embedding_dim = config.embedding_dim + self.pos_size
            pos_one_hot = np.eye(self.pos_size)
            one_hot_weight = torch.from_numpy(pos_one_hot).float()
            self.one_hot_embeddings = nn.Embedding(self.pos_size, self.pos_size, _weight=one_hot_weight)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            bidirectional=self.bidirectional,
                            num_layers=self.num_layers)

        if self.bidirectional:
            self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.tagset_size)
        else:
            self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        self.dropout = nn.Dropout(config.dropout)

    def init_hidden(self, tensor_size):
        if self.bidirectional:
            h0 = torch.zeros(2 * self.num_layers, tensor_size[1], self.hidden_dim)
            c0 = torch.zeros(2 * self.num_layers, tensor_size[1], self.hidden_dim)
        else:
            h0 = torch.zeros(self.num_layers, tensor_size[1], self.hidden_dim)
            c0 = torch.zeros(self.num_layers, tensor_size[1], self.hidden_dim)
        if self.device:
            h0 = h0.to(self.device)
            c0 = c0.to(self.device)
        return h0, c0

    def forward(self, X, y):
        X = self.word_embeddings(X)
        # Concatenate POS-embedding here
        if self.use_pos:
            POS = self.one_hot_embeddings(y)
            X = torch.cat((X, POS), dim=-1)
        X, _ = self.lstm(self.dropout(X))

        tag_space = self.hidden2tag(X.view(-1, X.shape[2]))
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores


class LSTMTagger(nn.Module):
    def __init__(self, config, dataloader):
        super(LSTMTagger, self).__init__()
        self.bidirectional = config.bidirection
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        self.hidden_dim = config.hidden_dim
        self.vocab_size = dataloader.vocab_size
        self.tagset_size = dataloader.tagset_size
        self.embedding_dim = config.embedding_dim
        self.device = config.device
        self.use_pos = config.use_pos

        if config.pretrained:
            self.word_embeddings = nn.Embedding.from_pretrained(dataloader.weights)
        else:
            self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        if self.use_pos:
            self.pos_size = dataloader.pos_size
            self.embedding_dim = config.embedding_dim + self.pos_size
            pos_one_hot = np.eye(self.pos_size)
            one_hot_weight = torch.from_numpy(pos_one_hot).float()
            self.one_hot_embeddings = nn.Embedding(self.pos_size, self.pos_size, _weight=one_hot_weight)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            bidirectional=self.bidirectional,
                            num_layers=self.num_layers)

        if self.bidirectional:
            self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.tagset_size)
        else:
            self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        self.dropout = nn.Dropout(config.dropout)

    def init_hidden(self, tensor_size):
        if self.bidirectional:
            h0 = torch.zeros(2 * self.num_layers, tensor_size[1], self.hidden_dim)
            c0 = torch.zeros(2 * self.num_layers, tensor_size[1], self.hidden_dim)
        else:
            h0 = torch.zeros(self.num_layers, tensor_size[1], self.hidden_dim)
            c0 = torch.zeros(self.num_layers, tensor_size[1], self.hidden_dim)
        if self.device:
            h0 = h0.to(self.device)
            c0 = c0.to(self.device)
        return h0, c0

    def forward(self, X, y):
        X = self.word_embeddings(X)
        # Concatenate POS-embedding here
        if self.use_pos:
            POS = self.one_hot_embeddings(y)
            X = torch.cat((X, POS), dim=-1)
        X, _ = self.lstm(self.dropout(X))

        tag_space = self.hidden2tag(X.view(-1, X.shape[2]))
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores


class CharLSTMTagger(nn.Module):
    def __init__(self, config, dataloader):
        super(CharLSTMTagger, self).__init__()
        self.dataloader = dataloader
        self.bidirectional = config.bidirection
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        self.hidden_dim = config.hidden_dim
        self.vocab_size = dataloader.vocab_size
        self.tagset_size = dataloader.tagset_size
        self.device = config.device
        self.char_embed_num = dataloader.char_vocab_size
        self.graph_embed_num = dataloader.graph_vocab_size
        self.char_dim = config.char_dim
        self.embedding_dim = config.embedding_dim + config.char_dim

        if config.char_pretrained:
            self.char_embeddings = nn.Embedding.from_pretrained(dataloader.char_weights)
            self.graph_embeddings = nn.Embedding.from_pretrained(dataloader.graph_weights)
        else:
            self.char_embeddings = nn.Embedding(self.char_embed_num, self.char_dim, padding_idx=0)
            self.graph_embeddings = nn.Embedding(self.graph_embed_num, self.char_dim, padding_idx=0)

            nn.init.xavier_uniform_(self.char_embeddings.weight)
            nn.init.xavier_uniform_(self.graph_embeddings.weight)

        self.char_level = config.use_char
        self.use_pos = config.use_pos

        if self.use_pos:
            self.pos_size = dataloader.pos_size
            self.embedding_dim = self.embedding_dim + self.pos_size
            pos_one_hot = np.eye(self.pos_size)
            one_hot_weight = torch.from_numpy(pos_one_hot).float()
            self.one_hot_embeddings = nn.Embedding(self.pos_size, self.pos_size, _weight=one_hot_weight)

        if config.pretrained:
            self.word_embeddings = nn.Embedding.from_pretrained(dataloader.weights)
        else:
            self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            bidirectional=self.bidirectional,
                            num_layers=self.num_layers)

        if self.bidirectional:
            self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.tagset_size)
        else:
            self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.tagset_size)

        self.dropout = nn.Dropout(config.dropout)
        self.dropout_embed = nn.Dropout(config.dropout_embed)

        # ------------CNN
        # Changed here for padding_idx error
        self.conv_filter_sizes = [3, 4]
        self.conv_filter_nums = self.char_dim  # 30
        self.convs = nn.ModuleList([
            nn.Conv3d(in_channels=1,
                      out_channels=self.conv_filter_nums,
                      kernel_size=(1, fs, self.char_dim))
            for fs in self.conv_filter_sizes
        ])

        self.fc = nn.Linear(len(self.conv_filter_sizes) * self.conv_filter_nums, self.char_dim)

    def tensortosent(self, tense):
        '''
            Returns the corresponding TEXT of given tensor
        '''
        return ' '.join([self.dataloader.txt_field.vocab.itos[i] for i in tense.cpu().data.numpy()])

    def get_char_tensor(self, X):
        word_int = []
        length = 0

        # Go through each tensor in each batch
        for b in range(0, X.shape[0]):
            each_X = X[b]
            char_int = []
            w = []

            # For character-level
            if self.char_level:
                # Get all the characters
                w += (list(x) for x in self.tensortosent(each_X).split())

                # Get all the index of those characters
                char_int += ([self.dataloader.char_field.vocab.stoi[c] for c in each] for each in w)

            # For grapheme-level
            else:
                w += (list(grapheme_clusters(x)) for x in self.tensortosent(each_X).split())
                char_int += ([self.dataloader.graph_field.vocab.stoi[c] for c in each] for each in w)

            if length < max(map(len, char_int)):
                length = max(map(len, char_int))

            word_int.append(char_int)

        # Padding to match the max_length words whose size is less than max(filter_size)
        if length < max(self.conv_filter_sizes):
            length += max(self.conv_filter_sizes) - length

        # Make each tensor equal in size
        X_char = np.array([[xi + [0] * (length - len(xi)) for xi in each] for each in word_int])

        # Convert to tensor from numpy array
        X_char = torch.from_numpy(X_char)

        return X_char

    # ---------------------------------------CHARACTER FORWARD
    def _char_forward(self, inputs):
        """
        Args:
            inputs: 3D tensor, [bs, max_len, max_len_char]
        Returns:
            char_conv_outputs: 3D tensor, [bs, max_len, output_dim]
        """
        max_len, max_len_char = inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)  # [bs, max_len * max_len_char]

        inputs = inputs.to(self.device)
        if self.char_level:
            input_embed = self.char_embeddings(inputs)  # [bs, ml*ml_c, feature_dim]
        else:
            input_embed = self.graph_embeddings(inputs)

        # Since convolution is 3-dimension, we need 5 dimension tensor
        input_embed = input_embed.view(-1, 1, max_len, max_len_char,
                                       self.char_dim)  # [bs, 1, max_len, max_len_char, feature_dim]

        # Convolution
        #         conved = [F.relu(conv(input_embed)).squeeze(3) for conv in self.convs]
        conved = [F.relu(conv(input_embed)) for conv in self.convs]

        # Pooling
        pooled = [torch.squeeze(torch.max(conv, -2)[0], -1) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        cat = cat.permute(0, 2, 1)

        return self.fc(cat)

    def init_hidden(self, tensor_size):
        if self.bidirectional:
            h0 = torch.zeros(2 * self.num_layers, tensor_size[1], self.hidden_dim)
            c0 = torch.zeros(2 * self.num_layers, tensor_size[1], self.hidden_dim)
        else:
            h0 = torch.zeros(self.num_layers, tensor_size[1], self.hidden_dim)
            c0 = torch.zeros(self.num_layers, tensor_size[1], self.hidden_dim)
        if self.device:
            h0 = h0.to(self.device)
            c0 = c0.to(self.device)
        return (h0, c0)

    def forward(self, X, y):
        X_char = self.get_char_tensor(X)
        X = self.word_embeddings(X)

        char_conv = self._char_forward(X_char)
        X = torch.cat((X, char_conv), dim=-1)

        # Concatenate POS-embedding here
        if self.use_pos:
            POS = self.one_hot_embeddings(y)
            X = torch.cat((X, POS), dim=-1)

        X, _ = self.lstm(self.dropout(X))
        tag_space = self.hidden2tag(X.view(-1, X.shape[2]))
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores
