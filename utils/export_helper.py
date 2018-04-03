import time

import tensorflow as tf


def export_serving_model(hparam, sess, encoder_index_input, decoder_index_input):
    """
    export serving model
    :param hparam:  hparam
    :param sess:  session
    :param encoder_index_input:  input tensor
    :param decoder_index_input:  output tensor
    :return:
    """
    output_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input': tf.saved_model.utils.build_tensor_info(encoder_index_input)},
            outputs={'output': tf.saved_model.utils.build_tensor_info(decoder_index_input)},
        )
    )

    builder = tf.saved_model.builder.SavedModelBuilder(hparam.export_serving_model_to + str(int(time.time())))

    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'chatbot_model': output_signature
        },
        legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op')
    )

    builder.save()


def export_model(hparams, model):
    if hparams.export_serving_model_to is None:
        raise Exception('export_serving_model_to must define')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(hparams.model_save_path))

        encoder_index_input = model['encoder_index_input']
        decoder_index_input = model['decoder_index_input']

        export_serving_model(hparams, sess, encoder_index_input, decoder_index_input)
