
# 定义开始标识
SOS_TOKEN = 0
# 定义结束标识
EOS_TOKEN = 1


class Lang(object):
    def __init__(self, lang):
        """
        定义word2sequence
        :param lang: 源语言和目标语言名字
        """
        self.lang = lang
        # 序列转文本
        self.id2char = {0 : '<SOS>', 1 : '<EOS>'}
        # 文本转序列
        self.char2id = {}


    def addSentence(self, sentence):
        """
        进行文本和序列的转换
        :param sentence: 输入的文本
        :return:
        """
        # 对文本进行遍历, 对每个词进行处理,外文中一般采用空格进行切分每个词
        for word in sentence.split(' '):
            self.addWord(word)


    def addWord(self, word):
        """
        详细处理函数
        :param word: 文本中的每个单词词
        :return:
        """
        if word not in self.char2id:
            self.id2char[len(self.id2char)] = word
        self.char2id = dict(zip(self.id2char.values(), self.id2char.keys()))


if __name__ == '__main__':
    engl = Lang('eng')
    sentence = 'hello I am Jay'
    engl.addSentence(sentence)
    print('id2char', engl.id2char)
    print('char2id', engl.char2id)
    print('vocab_size', len(engl.char2id))