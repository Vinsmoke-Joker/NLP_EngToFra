import torch.nn as nn
import torch
import torch.nn.functional as F


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AttnDecoderRnn(nn.Module):
    def __init__(self, output_size, hidden_size, dropout = 0.1, max_len = 10):
        """
        带有Attn的decoder
        :param hidden_size: [1, 1, hidden_size]
        :param output_size: [1, len(output.chat2id)]
        :param dropout:
        """
        super(AttnDecoderRnn, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drouput = dropout
        self.max_len = max_len

        self.embedding = nn.Embedding(
            num_embeddings = self.output_size,
            embedding_dim = self.hidden_size
        )
        # 根据attention计算方式,需要Q,K,V 三个输入参数
        # 常见计算方式有三种
        # 1.Q,K沿纵轴拼接,做一次线性变换后经过softmax 与V 作矩阵乘法
        # 2.Q,K沿着纵轴拼接,做一次线性变换后,经过tanh,内部求和后经softmax与V矩阵乘法
        # 3.Q与K的转置相乘，经过sotamax后 除以一个缩放系数与V 做矩阵乘法,系数为sqrt(embedding_dim)
        # 若是拼接,则以上计算结果需与Q再做一次拼接；若是第3中计算方法则不需要拼接,第三种方式一般为self_attn
        # 最后根据以上计算结果,经过线性变换得到相应的输出维度
        # 这里采用第一种方式进行计算
        # Q为decoer_input [1, 1] ,K 为prev_hidden 第一个时间步为encoder_hidden[1, 1, hidden_size]
        # Q经过embedding后[1, 1, embedding_dim] 这里embedding_dim==hidden_size 因此Q为[1, 1, hidden_size]
        # V为encoder_outputs,encoder中的输出为每个词的output,经过seq_len次的迭代得到得到输入句子的完整encoder_outputs,
        # encoder_outputs[max_len, hidden_size],在定义encoder_outputs时候,人为将句子长度限定在max_len的长度
        # Q,K经过拼接后, 输入维度为2 * hidden_size,由于要与V做矩阵乘法,输出维度为max_len->[1, 1, max_len]
        self.attn = nn.Linear(2 * self.hidden_size, self.max_len)
        # 由于是拼接操作,需要再一次与Q进行拼接,得到[1, 1, hidden_size * 2]
        # 拼接后的结果需要输入GRU中，GRU的input_size = hidden_size
        # 因此需要进行线性变换,输入维度为hidden_size * 2, 输出维度为hidden_size
        self.attn_combined = nn.Linear(2 * self.hidden_size, self.hidden_size)
        # 实例化一个dropout层
        self.drouput = nn.Dropout(p = self.drouput)
        self.gru = nn.GRU(
            input_size = self.hidden_size,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True,
            bidirectional = False
        )
        # 按照指定维度输出
        self.fc = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        """
        :param decoder_input: [batch_size ,1] ->[1, 1]
        :param decoder_hidden: [1, 1, hidden_size]
        :param encoder_outputs:  [max_len, hidden_size]
        :return:
        """
        input_embeded = self.embedding(decoder_input)
        # dropout 防止过拟合
        input_embeded = self.drouput(input_embeded)
        # 进行attention计算
        attn_weights = F.softmax(self.attn(torch.cat((input_embeded, decoder_hidden), dim = -1)), dim = -1)
        attn_weights_bmm = torch.bmm(attn_weights, encoder_outputs.unsqueeze(0))
        # 与Q进行拼接
        attn_combined = self.attn_combined(torch.cat((attn_weights_bmm, input_embeded), dim = -1))
        # 进行relu
        attn_combined = F.relu(attn_combined)
        # 进行GRU
        output, hidden = self.gru(attn_combined, decoder_hidden)
        # 进行输出
        return F.log_softmax(self.fc(output), dim = -1), hidden, attn_weights


    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)