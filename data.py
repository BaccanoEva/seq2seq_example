import re
import codecs
import collections
import numpy as np

class PTBreader(object):
    def __init__(self,source_path,target_path,batch_size,max_time_step):
        self.source_path = source_path
        self.target_path  = target_path
        self.sym_split = "([ .,!?\"':;)(])"
        self._WORD_SPLIT = re.compile(self.sym_split)
        self._DIGIT_RE = re.compile(br"\d")
        self.batch_size=batch_size
        self.max_time_step = max_time_step
        self.sentence_length = 0
        self._UNKnow = "<UnKonw>"
        self._PAD = "<PAD>"
        self._EOS = "EOS"
        self._START = "<S>"

        self.get_vocabary()
    #TODO:这里一次性读入，在大文件时会出问题
    def build_words_set(self,path):
        count = 0
        with codecs.open(path,"r","utf") as fp:
            words=[]
            for line in fp:
                space_separated_fragment = line.strip()
                tmp_set = self._WORD_SPLIT.split(space_separated_fragment)
                words.extend(tmp_set)
                count+=1
            self.sentence_length = count
            return words
    def build_vocabary(self,data):
        data_set =collections.Counter(data )
        delete_list=  ['.' ,'' ,':' ,',' ,' ',';']
        for item in delete_list:
            data_set.pop(item)
        count_pairs = sorted(data_set.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        # here we plus 4 is that we need to add START END falg
        word_to_id = dict(zip(words, range(4,len(words)+4)))
        return word_to_id
    def get_vocabary(self):
        source_data = self.build_words_set(self.source_path)
        target_data  = self.build_words_set(self.target_path)
        source_voca = self.build_vocabary(source_data)
        target_voca = self.build_vocabary(target_data  )
        source_voca[self._UNKnow] = 3
        target_voca[self._UNKnow] = 3
        source_voca[self._EOS] = 2
        target_voca[self._EOS] = 2
        source_voca[self._START] = 1
        target_voca[self._START] = 1
        source_voca[self._PAD]  = 0
        target_voca[self._PAD]  = 0

        self.START_ID = target_voca[self._START]
        self.EOS_ID = target_voca[self._EOS]
        self.id_src_vocabuary = {v: k for k, v in source_voca.items()}
        self.id_des_vocabuary = {v: k for k, v in target_voca.items()}
        self.source_voca = source_voca
        self.target_voca = target_voca

    def word_to_id(self,word,vocabuary):
        if word not in vocabuary:
            return vocabuary[self._UNKnow]
        else:
            return vocabuary[word]
    def id_to_word(self,word,vocabuary):
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
    def NextBatch(self):
        source =  codecs.open(self.source_path,"r","utf")
        target =  codecs.open(self.target_path,"r","utf")
        origin_src = ['None' for i in  range(self.batch_size)]
        origin_des = ['None' for i in  range(self.batch_size)]
        while True:
            for i in range(self.batch_size):
                    origin_src[i] =  source.readline()
                    origin_des[i] =  target.readline()
            yield (self.sentence_to_id(origin_src,self.source_voca),self.sentence_to_id(origin_des,self.target_voca))

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
