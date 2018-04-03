# Tensorflow Chatbot
A tensorflow chatbot use seq2seq model. 

# Functions
- [x] train a seq2seq model.
- [x] inference some sentence
- [x] export `tensorflow serving` model
- [x] command line support
- [ ] support read train data from file
- [ ] support chinese word split using `jieba`
- [ ] support summary for tensorboard 
- [ ] translate chinese comment to english
- [ ] add Django web support
- [ ] add http chatbot api
- [ ] add ui for chatbot


# Usage:
Train: `python main.py --mode=train`

Inference: `python main.py --mode=inference`

Export: `python main.py --mode=inference`

## Params:

| Param                   | default value          | Help                                                      |
| ----------------------- | ---------------------- | --------------------------------------------------------- |
| SYMBOLS_START           | `<s>`                  |                                                           |
| SYMBOLS_UNKNOWN         | `<unk>`                |                                                           |
| SYMBOLS_PAD             | `<pad>`                |                                                           |
| enc_sentence_length     | `10`                   | Length of encoder sentence                                |
| dec_sentence_length     | `10`                   | Length of decoder sentence                                |
| batch_size              | `4`                    | Batch_size                                                |
| n_epoch                 | `5000`                 | Epoch of train                                            |
| hidden_size             | `30`                   | Hidden size of rnn cell                                   |
| enc_emb_size            | `30`                   | embedding size of embedding_rnn_seq2seq                   |
| dec_emb_size            | `30`                   |                                                           |
| enc_vocab_size          | `None`                 | size of encoder vocab                                     |
| dec_vocab_size          | `None`                 | size of decoder vocab                                     |
| model_save_path         | `./model-checkpoints/` | path of model save to                                     |
| model_save_name         | `model.ckpt`           | name of saved model                                       |
| mode                    | `mode`                 | model of this program, must in (train, inference, export) |
| export_serving_model_to | `./serving_model/`     | export serving model to                                   |