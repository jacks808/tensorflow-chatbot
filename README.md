# Tensorflow Chatbot
A tensorflow chatbot project. use [`embedding_rnn_seq2seq`](https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/embedding_rnn_seq2seq) as the neural network model. 

![](https://ws4.sinaimg.cn/large/006tNc79ly1fq0re9uzhoj31ee0vu4dp.jpg)

# Functions

- [x] Support Not English data
- [x] Train a seq2seq model.
- [x] Support `Train`, `Inference`, `Export Serving Model` by different argument.
- [x] Inference some sentence
- [x] Auto save trained model, see param of: `model_save_path` and `model_save_name`.
- [x] Command line support
- [x] Support read train data from file
- [x] Support chinese word split using `jieba`
- [ ] Support summary for tensorboard 
- [ ] Translate chinese comment to english
- [ ] Add Django web support
- [ ] Add http chatbot api
- [ ] Add ui for chatbot


# Usage:
This project supoort different goal: cut a sentences file use `jieba`, train a model, inference a result or export your model for [tensorflow serving](https://www.tensorflow.org/serving/)

## Define data

Here we use a txt file to save train data. The data file contains a `question` and `answer` pre line.
Where line format is: `[question] | [answer]` e.g. `你好 | Hello`

The train data save at: `data/train.txt`. or set start argument: `--train_data_path='your_path_of_train_data'` to point out your data location.


## How to Use 
### Prepare Data: 

> `python main.py --mode=cut_data --data_path="your_data_path" --stopwords_path="stopword_path" --jieba_dict_path="dict_path"`

### Train:

> `python main.py --mode=train --data_path="your_data_path" --stopwords_path="stopword_path" --jieba_dict_path="dict_path"`

### Inference: 

> `python main.py --mode=inference --question="放个屁" --data_path="your_data_path" --stopwords_path="stopword_path" --jieba_dict_path="dict_path"`

### Export serving model: 

> `python main.py --mode=export --data_path="/Users/keen/workspace/seq2seq-chatbot/data/train.txt"`

## Params:

Parameters of this program is shown below:

| Param                   | default value          | Help                                                      |
| :---------------------- | ---------------------- | --------------------------------------------------------- |
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
| data_path               | `None`                 | train_data_path                                           |
| cut_data_postfix        | `.cut.txt`             | cut file postfix                                          |
| stopwords_path          | `None`                 | stop word file path                                       |
| jieba_dict_path         | `None`                 | jieba dict file path                                      |
| question                | `None`                 |                                                           |

# Update log

This project will update untill it ready to be a production

## 2018-04-04

* Add read data from file support

* Add `cut_data` mode

* Add `jieba` for support cut sentence

* Add `logconfig` at `main.py`

* Add arguments: 

  * `cut_data_postfix`: a cut train data file, generate by cut mode
  * `stopwords_path`: stop word file path
  * `jieba_dict_path`:jieba dict file path
  *  `question`: question for inference

* Modify argument: 

  *  `train_data_path` to `data_path`

* Update default hparam: 

  * `dec_sentence_length` from  `10` to `50` for support longer answer sentence
  *  `batch_size` from `6` to `10` for make train faster

  ​