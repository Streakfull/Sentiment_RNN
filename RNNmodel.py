from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
from string import punctuation
import torch.nn as nn


class RNNmodel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, lstm_dropout=0, layer_dropout=0, layers=1):
        super(RNNmodel, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, dropout=lstm_dropout,
                            num_layers=layers, batch_first=True)
        self.dropout = nn.Dropout(layer_dropout)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.test = nn.Linear(200, 1)

    def forward(self, input, hidden):
        input = input.long()
        embedding = self.embedding(input)
        lstm_output, _ = self.lstm(embedding, hidden)
        drop_out = self.dropout(lstm_output)
        output = self.output_layer(drop_out)
        output = self.sigmoid(output)
        # last prediction after the sequence cell
        return output[:, -1, :]

    def init_hidden(self, batch_size):
        return (
            nn.Parameter(torch.zeros(
                self.layers, batch_size, self.hidden_size)),
            nn.Parameter(torch.zeros(
                self.layers, batch_size, self.hidden_size))
        )
