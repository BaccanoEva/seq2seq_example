# Introduce
This is an example for translating cs to en using sequence2sequence with attention_model.

tensorflow version: 1.2.1.

I will make the code more clear if I have spare time
# Usage
1. make a new folder named nmt_data
2. download dataset in [here](https://nlp.stanford.edu/projects/nmt/). In **Preprocessed Data** select Train.en,Train.vi , Test data and word Vocabularies(50K). **Move the data into nmt_data**
3. In commandline run
```
python main.py
```
# Data_set
After downloading the data,the result should like followings.

```
├── __pycache__
│   ├── data.cpython-36.pyc
│   ├── helpers.cpython-36.pyc
│   └── trainslate_use_attention.cpython-36.pyc
├── attention_model.py
├── data.py
├── helpers.py
├── main.py
├── nmt_data
│   ├── train.en
│   ├── train.vi
│   ├── tst2012.en
│   ├── tst2012.vi
│   ├── tst2013.en
│   ├── tst2013.vi
│   ├── vocab.en
│   └── vocab.vi
├── README.md
└── translate_atten_log_new_api
    ├── events.out.tfevents.1504578074.DESKTOP-RHDFOO3
    ├── events.out.tfevents.1504578210.DESKTOP-RHDFOO3
    ├── events.out.tfevents.1504578632.DESKTOP-RHDFOO3
    └── model
        ├── checkpoint
        ├── model.ckpt-0.data-00000-of-00001
        ├── model.ckpt-0.index
        ├── model.ckpt-0.meta
        ├── model.ckpt-5.data-00000-of-00001
        ├── model.ckpt-5.index
        └── model.ckpt-5.meta
```
# Reference
https://github.com/tensorflow/nmt

https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/1-seq2seq.ipynb
