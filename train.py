import torch
import time
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from lineToTensor import prepareData, tensorFromPair, tensorFromSentence
import random
import os


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(input_tensor, target_tensor, encoder, decoder_attn, encoder_optimizer, decoder_optimizer, criterion, max_len = 10):
    """
    :param input_tensor: 源语言输入张量 [seq_len + 1, 1]
    seq_len 为输入文本长度,并非为max_len, + 1是因为添加了EOS, max_len在生成encoder_outputs时候
    人为将句子长度统一为max_len
    :param target_tensor: [seq_len + 1, 1]
    :param encoder: encoder
    :param decoder_attn: decoder_attn
    :param encoder_optimizer: encoder_optimizer
    :param decoder_optimizer:decoder_optimizer
    :param criterion: 损失函数
    :param max_len: encoder_outputs[max_len, hidden_size]
    :return:
    """
    # 初始化encoder隐藏层
    encoder_hidden = encoder.init_hidden()
    # 训练时,需要将优化器梯度置0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # 获取源语言和目标语言的长度,训练时encoder和decoder 每次输入为一个词
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # 初始化一个encoder_outputs, 用于拼接每一time step的encoder_output [1, 1, hidden_size]
    encoder_outputs = torch.zeros(max_len, encoder_hidden.size(-1)).to(device)
    # 设置初始损失为0
    loss = 0
    # 进行encoder
    for ei in range(input_length):
        # input_tensor [seq_len , 1]
        # input_tensor[ei] ->[1]
        # encoder_hidden [1, 1, hidden_size]
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # 将每个time step的output 进行保存
        # encoder_outputs[ei] ->[hidden_size]
        # encoder_output[1, 1, hidden_size]
        # encoder_output[0, 0] ->[hidden_size]
        encoder_outputs[ei] = encoder_output[0, 0]

    # 进行decoder
    # 初始化一个[batch_size , 1]的SOS 作为第一个time step的input
    decoder_input = torch.LongTensor([[0]]).to(device)  # SOS_TOKEN 为0

    # 第一个时间步的decoder_hidden 为 encoder最后一个time step 的hidden
    decoder_hidden = encoder_hidden
    # # 初始化一个decoder_outputs 用于保存每个time step的decoder_output
    # decoder_outputs = torch.zeros(max_len, output_size)
    teacher_forcing = random.random() > 0.5
    # teacher_forcing 只决定下次输入以真实值或者预测值,与损失计算无关
    if teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, attn_weights = decoder_attn(decoder_input, decoder_hidden, encoder_outputs)
            # 损失计算
            # target[di] ->[1]
            # decoder_output[1, 1, output_size]
            # decoder_output.squeeze() ->[1, output_size]
            loss += criterion(decoder_output.squeeze(0), target_tensor[di])
            # 由于使用teacher_forcing,下一个时间步采用真实值作为输入
            decoder_input = target_tensor[di].unsqueeze(0)
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, attn_weights = decoder_attn(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output.squeeze(0), target_tensor[di])
            # 不使用teacher_forcing,下一时间步采用预测值作为输出
            value, index = torch.topk(decoder_output, k = 1)
            # 判断是否达到EOS_token 1
            if index.squeeze().item() == 1:
                break
            # decoder_output [1, 1, output_size]
            # value [1, 1, 1]
            """
            a = torch.randn(1, 1, 3)
            
            a
            Out[22]: tensor([[[-1.2505,  0.2538, -1.3255]]])
            value, index = a.topk(1)
            
            value
            Out[24]: tensor([[[0.2538]]])
            
            index
            Out[25]: tensor([[[1]]])
            """
            decoder_input = index.squeeze(0)

    # 进行反向传播
    loss.backward()
    # 更新参数
    encoder_optimizer.step()
    decoder_optimizer.step()

    # 返回平均损失
    return loss.item() / target_length


def timeSince(since):
    """
    统计时间函数
    :param since: 开始时间
    :return:
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s = s - m * 60
    return '%d m, %d s' % (m, s)


input_lang, target_lang, pairs = prepareData('eng', 'fra')


def trainIters(encoder, decoder, n_iters, learning_rate = 0.01):
    """
    批次训练函数
    :param encoder: encoder的实例,传入train
    :param decoder:  decoder的实例，传入train
    :param n_iters: 迭代次数
    :param learning_rate:  学习率
    :return:
    """
    # 打印轮次和保存绘图点轮次
    print_every = 1000
    plot_every = 100
    # 获取训练开始时间
    start = time.time()
    # 绘制损失曲线的数据保存列表
    plot_losses = []
    # 每个打印间隔日志的总损失
    print_loss_total = 0
    # 每个绘制间隔的总损失
    plot_loss_total = 0


    # 加载模型
    if os.path.exists('./model/model_encoder.pkl'):
        encoder.load_state_dict(torch.load('./model/model_encoder.pkl'))
        decoder.load_state_dict(torch.load('./model/model_decoder.pkl'))

    # 初始化encoder和decoder的优化器
    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)

    # 损失函数
    criterion = nn.NLLLoss()

    # 进行迭代训练
    for iter in range(1, n_iters + 1):
        # 从语言对中随机选择一组作为输入和输出作为input 和 target
        # train_pair (input_tensor, target_tensor)
        train_pair = tensorFromPair(input_lang, target_lang, random.choice(pairs))
        input_tensor = train_pair[0]  # [seq_len + 1, 1]
        target_tensor = train_pair[1] # [seq_len + 1, 1]
        # 从train中获取相应的loss
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        # 进行损失累加
        print_loss_total += loss
        plot_loss_total += loss
        # 打印间隔进行输出
        if iter % print_every == 0:
            # 打印间隔的平均损失
            print_loss_avg = print_loss_total / print_every
            # 将损失置0 统计下一个间隔
            print_loss_total = 0
            # 打印信息
            print('time:{}, iter{}/{}%, print_loss_avg:{:.4f}'.format(timeSince(start), iter, iter * 100/ n_iters, print_loss_avg))

            torch.save(encoder.state_dict(), './model/model_encoder.pkl')
            torch.save(decoder.state_dict(), './model/model_decoder.pkl')

        if iter % plot_every == 0:
            # 绘图间隔将平均损失保存进列表并置0
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # 训练结束绘制图像
    plt.figure()
    plt.plot(plot_losses)
    plt.xlabel('100 * iters')
    plt.ylabel('loss')
    plt.xticks(rotation = 45)
    plt.savefig('./loss.png')


def evaluate(encoder, decoder, sentence, max_len = 10):
    """
    评估
    :param encoder: encoder 实例
    :param decoder: decoder 实例
    :param sentence: 待评估的句子 用于获取input_tensor
    :param max_len: 最大长度
    :return:
    """
    # 加载模型
    if os.path.exists('./model/model_encoder.pkl'):
        encoder.load_state_dict(torch.load('./model/model_encoder.pkl'))
        decoder.load_state_dict(torch.load('./model/model_decoder.pkl'))
    with torch.no_grad():
        # input_tensor [seq_len + 1, 1]
        input_tensor = tensorFromSentence(input_lang, sentence)
        # 获取输出tensor的句子长度
        input_length = input_tensor.size(0)
        # 初始化encoder_hidden [1, 1, hidden_size]
        encoder_hidden = encoder.init_hidden()
        # 初始化encoder_outputs [max_len, hidden_size]
        encoder_outputs = torch.zeros(max_len, hidden_size).to(device)
        # 逐字放入encoder
        for ei in range(input_length):
            # input_tensor[ei] ->[1]
            # encoder_output [1, 1, hidden_size]
            # encoder_hidden [1, 1, hidden_size]
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        # 初始化decoder_input [1, 1]
        decoder_input = torch.LongTensor([[0]]).to(device)
        # encoder最后一个time step的hidden作为decoder 第一个time step的hidden
        decoder_hidden = encoder_hidden
        # 存储预测的词汇列表
        decoded_words = []
        # 初始化存储attention的张量
        # decoder中attention_weights [1, 1, max_len]
        # decoder_attention 每行存储attention_weights对应位置的值
        # decoder 要经历max_len次循环，因此decoder_attention形状为[max_len, max_len]
        decoder_attention = torch.zeros(max_len, max_len).to(device)
        # 进行decoder
        for di in range(max_len):
            # 获取每个时间步decoder的输出
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 存储每个time step的attn_weights
            decoder_attention[di] = attn_weights[0, 0]
            value, index = torch.topk(decoder_output, k = 1)
            if index.squeeze().item() == 1:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(target_lang.id2char[index.squeeze().item()])
            decoder_input = index.squeeze(0)
        # 结果返回decoed_words, 以及注意力张量
        return decoded_words, decoder_attention


def evaluateRandom(encoder, decoder, n = 3):
    """
    进行随机测试
    :param encoder:
    :param decoder:
    :param n: 测试次数
    :return:
    """
    for i in range(n):
        # 随机选择一条语言对
        pair = random.choice(pairs)
        print('input:> \n', pair[0])
        print('true translate:>\n', pair[1])
        # 通过evaluate获取预测结果
        words_list, attention = evaluate(encoder, decoder, pair[0])
        # 将结果拼接为句子
        sentence = ' '.join(words_list)
        print('pred result:<\n', sentence)
        # 绘制attention
        plt.imshow(attention.cpu().detach().numpy())
        plt.savefig('./pre%d_attn.png' % i)


if __name__ == '__main__':
    from encoder import EncoderRNN
    from decoder_attn import AttnDecoderRnn
    hidden_size = 256
    # EncoderRNN(input_size, hidden_size)  ->(2803,256)
    encoder = EncoderRNN(len(input_lang.char2id), hidden_size).to(device)
    # AttnDecoderRnn(output_size, hidden_size ) ->(4345,256)
    decoder_attn = AttnDecoderRnn(len(target_lang.char2id), hidden_size).to(device)
    n_iters = 40000
    trainIters(encoder, decoder_attn, n_iters)
    evaluateRandom(encoder, decoder_attn)