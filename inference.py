import tensorflow as tf

from utils import chatbot_helper


def infer(hparams, model, data_info, question):
    # model_info
    encoder_sentence_input = model['encoder_sentence_input']  # 原文
    decoder_sentence_input = model['decoder_sentence_input']  # 原文
    encoder_index_input = model['encoder_index_input']  # 经过词典编码后的索引值
    decoder_index_input = model['decoder_index_input']  # 经过词典编码后的索引值
    predictions = model['predictions']
    training_op = model['training_op']
    loss = model['loss']

    # data info
    enc_vocab = data_info['enc_vocab']
    dec_vocab = data_info['dec_vocab']
    enc_reverse_vocab = data_info['enc_reverse_vocab']
    dec_reverse_vocab = data_info['dec_reverse_vocab']
    input_batches = data_info['input_batches']
    target_batches = data_info['target_batches']

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(hparams.model_save_path))
        # 输入
        encoder_data = chatbot_helper.build_input(hparams, sentences=question, vocab=enc_vocab)
        decoder_data = chatbot_helper.build_output(hparams, sentences=[''], vocab=dec_vocab)

        batch_pred, loss_value = sess.run(
            [predictions, loss],
            feed_dict={
                encoder_index_input: encoder_data,
                decoder_index_input: decoder_data
            }
        )

        # 输出
        for pred in batch_pred:
            result = chatbot_helper.idx2sent(hparams, pred, reverse_vocab=dec_reverse_vocab)

        return result, loss_value
