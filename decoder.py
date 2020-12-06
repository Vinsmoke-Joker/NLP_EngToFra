import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Dencoder
prev_hidden(encoder_hidden)    input([batch_size, 1] * SOS)
                                    embedding
                                    relu
                        gru
        hidden                      output
                                    log_softmax(fc(output))              
"""


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(DecoderRNN, self).__init__()
        """
        :param output_size: 目标的词表大小
        :param hidden_size: hidden_size
        """
        self.vocab_size = output_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.hidden_size
        )
        self.gru = nn.GRU(
            input_size = self.hidden_size,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )
        # decoder中最后要获取每个词的输出概率
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        # gru中output经过log_softmax
        self.log_softmax = nn.LogSoftmax()


    def forward(self, x, hidden):
        """
        :param x: decoder_input [batch_size,1]的SOS, 即[1, 1]
        :param hidden: encoder的hidden,即[1, 1, hidden_size]
        :return:
        """
        x = self.embedding(x) # [1, 1, embedding_dim]
        x = F.relu(x)
        # output [1, 1, hidden_size]
        # hidden [1, 1, hidden_size]
        output, hidden = self.gru(x, hidden)
        # fc->output[0]:[1, hidden_size]->[1, output_size]
        return self.log_softmax(self.fc(output[0])), hidden


    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
