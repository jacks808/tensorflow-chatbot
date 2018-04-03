import codecs
import re
from collections import Counter

import numpy as np


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
    :return: a data info dict, contains: enc_vocab, dec_vocab, enc_reverse_vocab, dec_reverse_vocab, input_batches, target_batches
    """
    all_input_sentences, all_target_sentences = read_data_from_file(hparams)

    # encoder data
    enc_vocab, enc_reverse_vocab, enc_vocab_size = build_vocab(hparams, all_input_sentences)

    # decoder data
    dec_vocab, dec_reverse_vocab, dec_vocab_size = build_vocab(hparams, all_target_sentences,
                                                               is_target=True)

    # update hparam
    hparams.enc_vocab_size = enc_vocab_size
    hparams.dec_vocab_size = dec_vocab_size

    # padding batch data
    batch_pad_size = len(all_input_sentences) % hparams.batch_size
    if batch_pad_size > 0:
        all_input_sentences.extend(all_input_sentences[:hparams.batch_size - batch_pad_size])
        all_target_sentences.extend(all_target_sentences[:hparams.batch_size - batch_pad_size])

    data_info = {
        'enc_vocab': enc_vocab,
        'dec_vocab': dec_vocab,
        'enc_reverse_vocab': enc_reverse_vocab,
        'dec_reverse_vocab': dec_reverse_vocab,
        'input_batches': np.reshape(all_input_sentences, [-1, hparams.batch_size]),
        'target_batches': np.reshape(all_target_sentences, [-1, hparams.batch_size]),
    }
    return data_info


def read_data_from_file(hparams):
    """
    read data from file
    :param hparams: use hparams.train_data_path
    :return:
    """
    encoder_data = []
    decoder_data = []

    with codecs.open(hparams.train_data_path) as file:
        for line in file.readlines():
            try:
                question, answer = line.strip().split('|')
                question = question.strip()
                answer = answer.strip()
            except ValueError:
                raise Exception("read_data_from_file error while handle line : ", line,
                                "please fix your data and try again")
            encoder_data.append(question)
            decoder_data.append(answer)

    return encoder_data, decoder_data
