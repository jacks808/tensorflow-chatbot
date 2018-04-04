import codecs
import logging
import os
import re
from collections import Counter

import jieba
import numpy as np


def cut_file(hparams):
    """
    cut a file from sentence to words
    :param hparams: hparams
    :return: None
    """
    src_file_path = hparams.data_path
    target_file_path = src_file_path + hparams.cut_data_postfix
    stopwords_path = hparams.stopwords_path

    # load stopwords set
    stopword_set = set()
    with open(stopwords_path, 'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    output = open(target_file_path, 'w', encoding='utf-8')
    with open(src_file_path, 'r', encoding='utf-8') as content:
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = cut_sentence(hparams, line)
            for word in words:
                if word not in stopword_set:
                    output.write(word.strip() + ' ')
            output.write('\n')

            if (texts_num + 1) % 1000 == 0:
                logging.info("process %d line" % (texts_num + 1))

        logging.info("Total cut %d line" % (texts_num + 1))
    output.close()


def cut_sentence(hparams, sentence):
    """
    cut word
    :param hparams:
    :param sentence:
    :return:
    """
    jieba_dict_path = hparams.jieba_dict_path

    if jieba.get_dict_file().name != hparams.jieba_dict_path:
        jieba.set_dictionary(jieba_dict_path)

    words = jieba.lcut(sentence, cut_all=False)
    return words


def tokenizer(hparams, sentence):
    """
    切词工具, 后续替换成jieba分词
    # Example:
        pprint(tokenizer('Hello world?? "sdfs%@#%'))
    :param sentence: 输入的句子
    :return: 词list
    """
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
        tokens = tokenizer(hparams, sentence)
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

    # use cut file
    data_path = hparams.data_path + hparams.cut_data_postfix
    if not os.path.exists(data_path):
        raise Exception("cut file not exists, please run `python main.py --mode=cut_data` ")

    with codecs.open(data_path) as file:
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
