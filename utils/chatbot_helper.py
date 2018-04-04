from utils import data_helper


# from utils import hparam


def token2idx(word, vocab):
    """
    词换取index

    Example:
        for token in tokenizer('Nice to meet you!'):
            print(token, token2idx(token, enc_vocab))

    :param word: 要查询的index
    :param vocab: 词典
    :return: index , 如果word不在词典中, 则返回<unk>对应的index
    """
    try:
        return vocab[word]
    except KeyError:
        # 如果是因为出现次数太少而不在词典中的词, 则返回<unk>
        return vocab['<unk>']


def sent2idx(hparams, sent, vocab, max_sentence_length=None, is_target=False):
    """
    句子换取index

    # Enc Example
        print('Hi What is your name?')
        print(sent2idx('Hi What is your name?'))

    # Dec Example
        print('Hi this is Jaemin.')
        print(sent2idx('Hi this is Jaemin.', vocab=dec_vocab, max_sentence_length=dec_sentence_length, is_target=True))

    :param hparams: hparams
    :param sent: 句子
    :param vocab: 词典
    :param max_sentence_length: 最大句子长度
    :param is_target:
    :return:
    """
    tokens = data_helper.tokenizer(hparams, sent)
    current_length = len(tokens)

    if current_length > max_sentence_length:
        raise Exception("current sentence length %d greater than max_sentence_length %d, sentence: `%s`" % (
            current_length, max_sentence_length, sent))

    pad_length = max_sentence_length - current_length
    if is_target:
        return [0] + [token2idx(token, vocab) for token in tokens] + [1] * pad_length
    else:
        return [token2idx(token, vocab) for token in tokens] + [0] * pad_length, current_length


def idx2token(idx, reverse_vocab):
    """
    index换取词
    :param idx: index
    :param reverse_vocab: 反查表 @see chatbot.build_vocab
    :return: 词
    """
    return reverse_vocab[idx]


def idx2sent(hparams, indices, reverse_vocab):
    """
    索引list换取词list
    :param hparams: 超参数
    :param indices: 索引list
    :param reverse_vocab: 反查表 see chatbot.build_vocab
    :return: 使用空格拼接的词list
    """
    return "".join([idx2token(idx, reverse_vocab) for idx in indices]).replace(hparams.SYMBOLS_PAD, "").replace(
        hparams.SYMBOLS_START, "").strip()


def build_input(hparams, sentences, vocab):
    """
    build input from sentences.
    :param hparams: hparams
    :param sentences: sentence list
    :param vocab: vocab
    :return:
    """
    result = []
    for sentence in sentences:
        input_sentence_index, _ = sent2idx(
            hparams=hparams,
            sent=sentence,
            vocab=vocab,
            max_sentence_length=hparams.enc_sentence_length,
            is_target=False
        )
        result.append(input_sentence_index)
    return result


def build_output(hparams, sentences, vocab):
    """
    build output from sentences.
    :param hparams:  hparams
    :param sentences:  sentence list
    :param vocab:  vocab
    :return:
    """
    decoder_data = []
    for sentence in sentences:
        decoder_index = sent2idx(
            hparams=hparams,
            sent=sentence,
            vocab=vocab,
            max_sentence_length=hparams.dec_sentence_length,
            is_target=True)
        decoder_data.append(decoder_index)
    return decoder_data
