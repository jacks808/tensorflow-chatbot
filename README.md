# Tensorflow Chatbot
A tensorflow chatbot use seq2seq model. 

# Functions
- [x] train a seq2seq model.
- [x] inference some sentence
- [x] export `tensorflow serving` model
- [x] command line support
- [x] support read train data from file (finish at 2018-04-03)
- [ ] support chinese word split using `jieba`
- [ ] support summary for tensorboard 
- [ ] translate chinese comment to english
- [ ] add Django web support
- [ ] add http chatbot api
- [ ] add ui for chatbot


# Usage:
## Define data
Here we use a txt file to save train data. The data file contains a `question` and `answer` pre line.
Where line format is: `[question] | [answer]` e.g. `你好 | Hello`

The train data save at: `data/train.txt`. or set start argument: `--train_data_path='your_path_of_train_data'` to point out your data location.


## Run 
Train: `python main.py --mode=train`

Inference: `python main.py --mode=inference`

Export serving model: `python main.py --mode=inference`

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
| train_data_path         | `None`                 | train_data_path                                           |

