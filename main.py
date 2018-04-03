import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_seq2seq
from tensorflow.contrib.rnn import BasicRNNCell

import inference
import train
from utils import data_helper
from utils import export_helper

flags = tf.app.flags
# 字典补全相关
flags.DEFINE_string(name="SYMBOLS_START", default="<s>", help="开始字符")
flags.DEFINE_string(name="SYMBOLS_UNKNOWN", default="<unk>", help="未知字符")
flags.DEFINE_string(name="SYMBOLS_PAD", default="<pad>", help="补全字符")

# 编码字符长度
flags.DEFINE_integer(name="enc_sentence_length", default=10, help="enc_sentence_length")
flags.DEFINE_integer(name="dec_sentence_length", default=10, help="dec_sentence_length")
flags.DEFINE_integer(name="batch_size", default=4, help="batch_size")

# hparams of graph
flags.DEFINE_integer(name="n_epoch", default=5000, help="n_epoch")
flags.DEFINE_integer(name="hidden_size", default=30, help="hidden_size")

flags.DEFINE_integer(name="enc_emb_size", default=30, help="enc_emb_size")
flags.DEFINE_integer(name="dec_emb_size", default=30, help="dec_emb_size")
flags.DEFINE_integer(name='enc_vocab_size', default=None, help="编码词典大小")
flags.DEFINE_integer(name='dec_vocab_size', default=None, help="解码词典大小")

flags.DEFINE_string(name='model_save_path', default='./model-checkpoints/', help='model_save_path')
flags.DEFINE_string(name='model_save_name', default='model.ckpt', help='model_save_name')

flags.DEFINE_string(name='mode', default='train', help='mode of this program(train, inference, export)')

flags.DEFINE_string(name='export_serving_model_to', default='./serving_model/', help='export serving model to')
FLAGS = flags.FLAGS


def create_hparams(flags):
    return tf.contrib.training.HParams(
        # 补充符号
        SYMBOLS_START=flags.SYMBOLS_START,
        SYMBOLS_UNKNOWN=flags.SYMBOLS_UNKNOWN,
        SYMBOLS_PAD=flags.SYMBOLS_PAD,

        enc_sentence_length=flags.enc_sentence_length,
        dec_sentence_length=flags.dec_sentence_length,
        batch_size=flags.batch_size,

        n_epoch=flags.n_epoch,
        hidden_size=flags.hidden_size,

        enc_emb_size=flags.enc_emb_size,
        dec_emb_size=flags.dec_emb_size,
        enc_vocab_size=flags.enc_vocab_size,
        dec_vocab_size=flags.dec_vocab_size,

        model_save_path=flags.model_save_path,
        model_save_name=flags.model_save_name,

        mode=flags.mode,

        export_serving_model_to=flags.export_serving_model_to,

    )


def def_model(hparams):
    """
    build graph
    :return:
    """
    tf.reset_default_graph()
    encoder_index_input = tf.placeholder(  # shape = [?,10]
        tf.int64,
        shape=[None, hparams.enc_sentence_length],
        name='input_sentence')

    encoder_sentence_input = tf.placeholder(dtype=tf.string, name='input')
    # 使用端对端输入
    # encoder_index_input = tf.reshape(
    #     tf.py_func(
    #         func=chatbot_helper.build_input,
    #         inp=[hparams, encoder_sentence_input],
    #         Tout=[tf.int64]
    #     ),
    #     shape=[-1, hparams.enc_sentence_length])

    decoder_index_input = tf.placeholder(
        tf.int64,
        shape=[None, hparams.dec_sentence_length + 1],
        name='output_sentences')
    # batch_major => time_major
    enc_inputs_t = tf.transpose(encoder_index_input, [1, 0])
    dec_inputs_t = tf.transpose(decoder_index_input, [1, 0])
    rnn_cell = BasicRNNCell(hparams.hidden_size)

    # rnn_cell = LSTMCell(hidden_size) # work well
    with tf.variable_scope("embedding_rnn_seq2seq"):
        # dec_outputs: [dec_sent_len+1 x batch_size x hidden_size]
        dec_outputs, dec_last_state = embedding_rnn_seq2seq(
            encoder_inputs=tf.unstack(enc_inputs_t),  # a list
            decoder_inputs=tf.unstack(dec_inputs_t),  # a list
            cell=rnn_cell,
            num_encoder_symbols=hparams.enc_vocab_size + 2,  # +2因为补充了<s> <unk>
            num_decoder_symbols=hparams.dec_vocab_size + 3,  # +3因为补充了 <s> <unk> <pad>
            embedding_size=hparams.enc_emb_size,
            feed_previous=True)

    # predictions: [batch_size x dec_sentence_lengths+1]
    predictions = tf.transpose(tf.argmax(tf.stack(dec_outputs), axis=-1), [1, 0])

    # labels & logits: [dec_sentence_length+1 x batch_size x dec_vocab_size+2]
    labels = tf.one_hot(dec_inputs_t, hparams.dec_vocab_size + 3)
    logits = tf.stack(dec_outputs)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))

    # training_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    training_op = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss)

    model = {
        "encoder_sentence_input": encoder_sentence_input,  # 输入句子
        "decoder_sentence_input": encoder_sentence_input,  # 输出句子
        "encoder_index_input": encoder_index_input,  # 输入句子的索引
        "decoder_index_input": decoder_index_input,  # 输出句子的索引
        "predictions": predictions,  # 模型输出
        "loss": loss,
        "training_op": training_op
    }

    return model


def main(_):
    hparams = create_hparams(FLAGS)
    data_info = data_helper.init_data(hparams)
    model = def_model(hparams)

    if hparams.mode == 'train':
        train.train(hparams, model, data_info)
    elif hparams.mode == 'inference':
        inference.infer(hparams, model, data_info, '你好')
    elif hparams.mode == 'export':
        export_helper.export_model(hparams, model)
    else:
        raise Exception("mode error, must in (train, inference, export)")


if __name__ == '__main__':
    tf.app.run()
