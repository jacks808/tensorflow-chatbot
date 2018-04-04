import tensorflow as tf

from utils import chatbot_helper
from utils import export_helper


def train(hparams, model, data_info):
    # model info
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
        sess.run(tf.global_variables_initializer())
        # loss_history = []

        for epoch in range(hparams.n_epoch):

            all_preds = []
            epoch_loss = 0
            for input_batch, target_batch in zip(input_batches, target_batches):
                # Evaluate three operations in the graph => predictions, loss, training_op(optimzier)
                batch_preds, batch_loss, _ = sess.run(
                    [predictions, loss, training_op],
                    feed_dict={
                        encoder_index_input: chatbot_helper.build_input(hparams, input_batch, enc_vocab),
                        decoder_index_input: chatbot_helper.build_output(hparams, target_batch, dec_vocab)
                    })
                # loss_history.append(batch_loss)
                epoch_loss += batch_loss
                all_preds.append(batch_preds)

            # Logging every 400 epochs
            if epoch % 500 == 0:
                for input_batch, target_batch, batch_preds in zip(input_batches,
                                                                  target_batches,
                                                                  all_preds):
                    for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                        print('\tinput  :', input_sent,
                              '\n\tchatbot: ', chatbot_helper.idx2sent(hparams, pred, reverse_vocab=dec_reverse_vocab),
                              '\n\tCorrect:', target_sent, "\n")

                print('Epoch, %d\tepoch loss: %f\n' % (epoch, epoch_loss))

                # save model checkpoint
                saver.save(sess, hparams.model_save_path + hparams.model_save_name, global_step=epoch)

        if hparams.export_serving_model_to:
            export_helper.export_serving_model(hparams, sess, encoder_index_input, decoder_index_input)
