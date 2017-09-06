import re
import codecs
import collections
import numpy as np

class PTBreader(object):
    def __init__(self,
                src_train_path,des_train_path,src_test_path,des_test_path,
                src_dev_path,
                des_dev_path,
                src_vocab_path,
                des_vocab_path,
                batch_size,
                max_time_step):

        self.src_train_path = src_train_path
        self.des_train_path  = des_train_path

        self.src_test_path = src_test_path
        self.des_test_path  = des_test_path

        self.src_dev_path = src_dev_path
        self.des_dev_path  = des_dev_path

        self.src_vocab_path  = src_vocab_path
        self.des_vocab_path  = des_vocab_path

        self.batch_size=batch_size
        self.max_time_step = max_time_step
        # used when we need to convert id to words
        self._UNKnow = "<unk>"
        self._PAD = "<pad>" # we should manually add _pad id
        self._EOS = "</s>"
        self._START = "<s>"

        self._get_vocabary()
        self._getLenght()
        print('src_train_len:{} and  des_train_len:{}'.format(self.src_train_len,self.des_train_len))
        print('src_test_len:{} and  des_test_len:{}'.format(self.src_test_len,self.des_test_len))
        print('src_dev_len:{} and  des_dev_len:{}'.format(self.src_dev_len,self.des_dev_len))
        print('data read completed')

        self.train_batch_length = self.src_train_len // self.batch_size
        self.test_batch_length = self.src_test_len // self.batch_size
        self.dev_batch_length = self.src_dev_len // self.batch_size
    def _get_length(self,path):
        count = 0
        with codecs.open(path,"r","utf") as fp:
            for line in fp:
                count+=1
        return count
    def _getLenght(self):
        self.src_train_len = self._get_length(self.src_train_path)
        self.des_train_len = self._get_length(self.des_train_path)
        self.src_test_len = self._get_length(self.src_test_path)
        self.des_test_len = self._get_length(self.des_test_path)
        self.src_dev_len = self._get_length(self.src_dev_path)
        self.des_dev_len = self._get_length(self.des_dev_path)
    def _read_vocabary(self,path):
        voc = [self._PAD]
        with codecs.open(path,"r","utf") as fp:
            for line in fp:
                voc.append(line.strip('\n').strip('\r'))
        vocabuary = dict(zip(voc,range(0,len(voc))))
        return vocabuary
    def _get_vocabary(self):
        self.source_voca = self._read_vocabary(self.src_vocab_path)
        self.target_voca = self._read_vocabary(self.des_vocab_path)

        self.START_ID = self.target_voca[self._START]
        self.EOS_ID = self.target_voca[self._EOS]
        self.id_src_vocabuary = {v: k for k, v in self.source_voca.items()}
        self.id_des_vocabuary = {v: k for k, v in self.target_voca.items()}
    def word_to_id(self,word,vocabuary):
        if word not in vocabuary:
            return vocabuary[self._UNKnow]
        else:
            return vocabuary[word]
    def id_to_word(self,word,flag):
        if flag == 'src':
            vocabuary = self.id_src_vocabuary
        else:
            vocabuary = self.id_des_vocabuary
        if word not in vocabuary:
            return self._UNKnow
        else:
            return vocabuary[word]
    def sentence_to_id (self,sentences,vocabuary):
        sen_id=[]
        for i,sentence in enumerate(sentences):
            sentence  = [self.word_to_id(x,vocabuary) for x in sentence.split()]
            sen_id.append(sentence)
        return sen_id
    def NextBatch(self,usage='train'):
        if usage is 'train':
            src_path = self.src_train_path
            des_path = self.des_train_path
        elif usage is 'test':
            src_path = self.src_test_path
            des_path = self.des_test_path
        elif usage is 'dev':
            src_path = self.src_dev_path
            des_path = self.des_dev_path
        else:
            print('ERROR.Please input set')
        source =  codecs.open(src_path,"r","utf")
        target =  codecs.open(des_path,"r","utf")
        origin_src = ['None' for i in  range(self.batch_size)]
        origin_des = ['None' for i in  range(self.batch_size)]
        while True:
            for i in range(self.batch_size):
                    origin_src[i] =  source.readline().strip('\r')
                    origin_des[i] =  target.readline().strip('\r')
            yield (self.sentence_to_id(origin_src,self.source_voca),self.sentence_to_id(origin_des,self.target_voca))

    """batch the input list into a sized np.array to feed the data """
    def batch(self,inputs, max_sequence_length=None):
        if  max_sequence_length:
            sequence_lengths=[]
            for seq in inputs:
                if len(seq)>=max_sequence_length:
                    sequence_lengths.append(max_sequence_length)
                else:
                    sequence_lengths.append(len(seq))
        else:
            sequence_lengths = [len(seq) for seq in inputs]

        batch_size = len(inputs)

        if max_sequence_length >max(sequence_lengths):
            max_sequence_length = max(sequence_lengths)
        else:
            pass

        inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
        for i, seq in enumerate(inputs):
            for j, element in enumerate(seq):
                if j >= max_sequence_length:
                    break
                else:
                    inputs_batch_major[i, j] = element

        # [batch_size, max_time] -> [max_time, batch_size]
        inputs_time_major = inputs_batch_major.swapaxes(0, 1)

        return inputs_time_major, sequence_lengths
    def next_feed(self,model,idx,idy):
        encoder_inputs_, input_length = self.batch(idx,self.max_time_step)
        decoder_targets_,target_length = self.batch(
            [(sequence) + [self.EOS_ID] for sequence in idy],self.max_time_step
        )
        decoder_inputs_, decode_length= self.batch(
            [[self.START_ID] + (sequence) for sequence in idy],self.max_time_step
        )
        return {
            model.encoder_inputs: encoder_inputs_,
            model.decoder_inputs: decoder_inputs_,
            model.decoder_targets: decoder_targets_,
            model.input_sequence_length:input_length,
            model.target_sequence_length:target_length,
            model.decoder_sequence_length:decode_length
        }
