from load_data import readLangs, filtersPairs
import random
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def prepareData(lang1, lang2):
    """
    完成对读取数据的字符映射和过滤
    :param lang1:
    :param lang2:
    :return:
    """
    input_lang, target_lang, pairs = readLangs(lang1, lang2)
    pairs = filtersPairs(pairs)
    # [['i m .', 'j ai ans .'], ['i m ok .', 'je vais bien .']...]
    for pair in pairs:
        # pair ['i m .', 'j ai ans .']
        # pair[0] ['i m .']
        input_lang.addSentence(pair[0])
        target_lang.addSentence(pair[1])
    return input_lang, target_lang, pairs


def tensorFromSentence(lang, sentence):
    """
    对输入句子进行序列化
    :param lang: 语言类型
    :param sentence: 文本经过处理后的pair
    :return:
    """
    indexes = [lang.char2id[line] for line in sentence.split(' ')]
    # 在语言后加入EOS 标识 <EOS> : 1
    indexes.append(1)
    # 封装为tensor返回,并改变形状为[-1, 1]
    return torch.LongTensor(indexes).view(-1, 1).to(device)


def tensorFromPair(input_lang, target_lang, pair):
    """
    分别处理input和target
    :param pair: 语言对
    :param input_lang : 源语言词表对象
    :param target_lang : 目标语言词表对象
    :return:
    """
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(target_lang, pair[1])
    return (input_tensor, target_tensor)


if __name__ == '__main__':
    input_lang, target_lang, pairs = prepareData('eng', 'fra')
    print('input_lang_vocab', len(input_lang.char2id)) #
    print('target_lang_vocab', len(target_lang.char2id)) #
    print(random.choice(pairs))
    pair_tensor = tensorFromPair(input_lang, target_lang, pairs[0])
    print(pair_tensor)
    print('input_tensor_size', pair_tensor[0].size())
    print('target_tensor_size', pair_tensor[1].size())
    print('pair_tensor[0][0].size()', pair_tensor[0][0].size())

