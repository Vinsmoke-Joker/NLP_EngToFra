from pre_process import normalizeString
from word2sequence import Lang

data_path = './data/eng-fra.txt'
MAX_LEN = 10
# 选择带有指定前缀的语言特征数据作为训练数据
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def readLangs(lang1, lang2):
    """
    读取源语言和目标语言,规范化处理后,写入input和target文件
    :param lang1: 源语言,用于构建源语言词表对象
    :param lang2: 目标语言,用于目标源语言词表对象
    :return:
    """
    with open(data_path, encoding = 'utf-8') as f:
        # Go.\tVa !  即 源语言\t目标语言
        # readlines()->'Go.\tVa !\n'
        content = f.read().strip().split('\n')
        # 按\n拆分后[Go.\tVa !, Go.\tVa !]
        # 对源语言目标语言按\t拆分后[[GO., Va !],[[GO., Va !]],分别进行规范化处理
        # line 为\n拆分后列表
        # li 为\t后 [[GO., Va !],[[GO., Va !]]的子列表 [GO., Va !]
        pairs = [[normalizeString(li) for li in line.split('\t')] for line in content]
        # pairs [[规范化后的源语言,规范化后的目标语言], [规范化后的源语言,规范化后的目标语言]]
        # [['go .', 'va !'], ['run !', 'cours !'], ['run !', 'courez !']]
        input_lang = Lang(lang1)
        target_lang = Lang(lang2)
        return input_lang, target_lang, pairs


def filterPair(pair):
    """
    对语言对进行过滤
    :param pair: 读取文件后,生成的语言对
    :return:
    """
    # 经过预处理的句子,词与词之间均以空格进行分隔
    return len(pair[0].split(' ')) < MAX_LEN and \
            pair[0].startswith(eng_prefixes) and \
            len(pair[1].split(' ')) < MAX_LEN


def filtersPairs(pair):
    return list(filter(lambda x : filterPair(x), pair))


if __name__ == '__main__':
    input_lang, target_lang, pairs = readLangs('eng', 'fra')
    print(pairs[:5])
    res = filtersPairs(pairs)
    print('filter_res:', res[:5])