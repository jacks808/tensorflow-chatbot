import re
from collections import Counter


# from utils import hparam

# input_batches = [
#     ['Hi What is your name?', 'Nice to meet you!'],
#     ['Which programming language do you use?', 'See you later.'],
#     ['Where do you live?', 'What is your major?'],
#     ['What do you want to drink?', 'What is your favorite beer?']]
#
# target_batches = [
#     ['Hi this is Jaemin.', 'Nice to meet you too!'],
#     ['I like Python.', 'Bye Bye.'],
#     ['I live in Seoul, South Korea.', 'I study industrial engineering.'],
#     ['Beer please!', 'Leffe brown!']]


def def_input_batches():
    input_batches = [
        ['你 叫 什么?', '很高兴 认识 你!'],
        ['你 多大 了?', '再见.'],
        ['你 住在 哪?', '你是 做什么 的?'],
        ['你会 什么?', '最 擅长 什么?'],
        ['在吗?', '你 喜欢 我 吗?'],
        ['吃了吗?'],
        ['你好'],
        ['你是 谁?'],
    ]
    return input_batches


def def_target_batches():
    target_batches = [
        ['我是 小黄鸡', '我也 很高兴 认识 你!'],
        ['年龄 是个 秘密!', 'Bye Bye.'],
        ['我 住在 中国 的 硅谷, 西二旗!', '我 负责 陪你 聊天'],
        ['琴棋书画 样样精通!', '被 调教!'],
        ['我在 我在', '就 那样 吧'],
        ['没吃, 你要 请 我去 WJ 咖啡 吗?'],
        ['恩 就 那样 吧'],
        ['小黄鸡'],
    ]
    return target_batches


# all_input_sentences = []
# for input_batch in input_batches:
#     all_input_sentences.extend(input_batch)

# all_target_sentences = []
# for target_batch in target_batches:
#     all_target_sentences.extend(target_batch)
#
# pprint("输入字典:\n")
# pprint(all_input_sentences)
# pprint("输出字典:\n")
# pprint(all_target_sentences)


def tokenizer(sentence):
    """
    切词工具, 后续替换成jieba分词
    # Example:
        pprint(tokenizer('Hello world?? "sdfs%@#%'))
    :param sentence: 输入的句子
    :return: 词list
    """
    # print(type(sentence))
    if isinstance(sentence, bytes):
        sentence = sentence.decode("utf-8")

    tokens = re.findall(r"[\w]+|[^\s\w]", sentence)
    return tokens


def build_vocab(hparams, sentences, is_target=False, max_vocab_size=None):
    """
    生成词典

    # Example:
        pprint(build_vocab(all_input_sentences))
        print('\n')
        pprint(build_vocab(all_target_sentences))

    :param hparams: hparams
    :param sentences: 句子(不需要切词)
    :param is_target: 是否为decoder使用
    :param max_vocab_size: 最大词典大小
    :return: 词典(使用词查id), 反查表(使用id查词), 词典大小
    """
    # 获取counter
    word_counter = Counter()
    vocab = dict()
    reverse_vocab = dict()

    # 遍历sentences, 并进行切词和统计
    for sentence in sentences:
        tokens = tokenizer(sentence)
        word_counter.update(tokens)

    # 确定词典大小
    if max_vocab_size is None:
        max_vocab_size = len(word_counter)

    # 如果是解码的句子, 则补充开始符号: <s> 和 补全符号<pad>
    if is_target:
        vocab[hparams.SYMBOLS_START] = 0
        vocab[hparams.SYMBOLS_PAD] = 1
        vocab[hparams.SYMBOLS_UNKNOWN] = 2
        vocab_idx = 3
        for key, value in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_idx
            vocab_idx += 1
    else:
        vocab[hparams.SYMBOLS_PAD] = 0
        vocab[hparams.SYMBOLS_UNKNOWN] = 1
        vocab_idx = 2
        for key, value in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_idx
            vocab_idx += 1

    # 生成反查表
    for key, value in vocab.items():
        reverse_vocab[value] = key

    # 返回: 词典(使用词查id), 反查表(使用id查词), 词典大小
    return vocab, reverse_vocab, max_vocab_size


def init_data(hparams):
    """
    init data
    :param hparams:
    :return:
    """
    # encoder数据
    input_batches = def_input_batches()
    all_input_sentences = []
    for input_batch in input_batches:
        all_input_sentences.extend(input_batch)
    enc_vocab, enc_reverse_vocab, enc_vocab_size = build_vocab(hparams, all_input_sentences)

    # decoder数据
    target_batches = def_target_batches()
    all_target_sentences = []
    for target_batch in target_batches:
        all_target_sentences.extend(target_batch)
    dec_vocab, dec_reverse_vocab, dec_vocab_size = build_vocab(hparams, all_target_sentences,
                                                                           is_target=True)

    # update hparam
    hparams.enc_vocab_size = enc_vocab_size
    hparams.dec_vocab_size = dec_vocab_size
    data_info = {
        'enc_vocab': enc_vocab,
        'dec_vocab': dec_vocab,
        'enc_reverse_vocab': enc_reverse_vocab,
        'dec_reverse_vocab': dec_reverse_vocab,
        'input_batches': input_batches,
        'target_batches': target_batches,

    }
    return data_info