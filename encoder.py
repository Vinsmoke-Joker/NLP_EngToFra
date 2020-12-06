import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


"""
Dencoder
prev_hidden                          input
                                    embedding
                        gru
        hidden                      output             
"""


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        :param input_size: vocab_size len(char2id)
        :param hidden_size:
        """
        super(EncoderRNN, self).__init__()
        self.vocab_size = input_size
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
            bidirectional = False
        )


    def forward(self, x, hidden):
        """
        :param x: x形状为[1]
        input_tensor由于添加了EOS,所有文本长度为原文本长度max_len + 1,
        因此input_tensor为[max_len + 1, 1] 该tensor每一行为一个词的id映射tensor
        在训练过程中,每次取一个词放入encoder中,共取max_len + 1即文本长度次
        :param hidden: 初始化的hidden [1, 1, hidden_size]
        :return:
        """
        # x:[1]->embedding:[1, embedding]->gru输入要求，unsqueeze(0)->[1, 1, embedding_dim]
        x = self.embedding(x).unsqueeze(0)
        # output:[1, 1, embedding_dim] -> [1, 1, hidden_size]
        # hidden:[1, 1, hidden_size]
        output, hidden = self.gru(x, hidden)
        return output, hidden


    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)

