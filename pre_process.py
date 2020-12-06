import unicodedata
import re


# 字符规范化-去除重音等
def unicodeAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD' ,s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """
    :param s: 传入的单词
    :return:
    """
    s = unicodeAscii(s.lower().strip())
    # 对!?.等前面加一个空格
    s = re.sub(r'([!?.])', r' \1', s)
    # 将不是大小写字母和正常标点的都替换为空格
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return  s


if __name__ == '__main__':
    s = 'Are u kidding me ?'
    c = normalizeString(s)
    print(c)  # are u kidding me ?