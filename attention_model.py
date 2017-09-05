import  tensorflow as tf
import data
import os
import argparse
from tensorflow.python.layers import core as layers_core
import time
def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # network
    parser.add_argument("--model", type=str, default="train", help="used for what purpose:train,evalate or see .")

class Seq2seq(object):
    def __init__(self,max_time_step,batch_size,encoder_hidden_units,src_vocab,des_vocab,input_embedding_size
                 ,decoder_hidden_units,learning_rate,
                 is_trainging,
                 tgt_sos_id,tgt_eos_id,
                 layer = None,
                 max_gradient_norm=1,
                 initial_learning_rate=0.0001
                 ):
        self.max_time_step = max_time_step
        self.batch_size = batch_size
        self.encoder_hidden_units=encoder_hidden_units
        self.src_vocab = src_vocab
        self.des_vocab = des_vocab
        self.src_vocab_size = len(src_vocab)
        self.des_vocab_size = len(des_vocab)
        self.input_embedding_size = input_embedding_size
        self.decoder_hidden_units = decoder_hidden_units
        self.initial_learning_rate = learning_rate
        self.is_trainging = is_trainging
        self.tgt_sos_id = tgt_sos_id# the start of the sentence
        self.tgt_eos_id = tgt_eos_id# the end of the sentence
        self.max_gradient_norm = max_gradient_norm
        self.initial_learning_rate= initial_learning_rate
    def create_train_model(self):
        """input and target"""
        self.encoder_inputs = tf.placeholder(shape=(None, self.batch_size), dtype=tf.int32, name='encoder_inputs')
        self.decoder_targets = tf.placeholder(shape=(None, self.batch_size), dtype=tf.int32, name='decoder_targets')
        self.decoder_inputs = tf.placeholder(shape=(None, self.batch_size), dtype=tf.int32, name='decoder_inputs')
        self.input_sequence_length = tf.placeholder(shape=([self.batch_size]), dtype=tf.int32, name='input_length')
        self.decoder_sequence_length = tf.placeholder(shape=([self.batch_size]), dtype=tf.int32, name='decoder_inputs_length')
        self.target_sequence_length = tf.placeholder(shape=([self.batch_size]), dtype=tf.int32, name='target_sequence_length')

        self.encoder_embeddings = tf.get_variable('encoder_embedding', [self.src_vocab_size, self.input_embedding_size],
                                             initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                             dtype=tf.float32)
        self.decoder_embeddings = tf.get_variable('decoder_embedding',[self.des_vocab_size, self.input_embedding_size],
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                                                 dtype=tf.float32)
        """set the decay learning_rate"""
        # TODO: I don't konw whether to keep this learning_rate when inference because there may meet errors
        # when I want to restore the Variable but the Variable learning_rate don't exist
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=500,decay_rate=0.9)
        self.add_global = global_step.assign_add(1)
        """Embedding"""
        encoder_inputs_embedded = tf.nn.embedding_lookup(self.encoder_embeddings, self.encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(self.decoder_embeddings, self.decoder_inputs)

        """Encoder"""
        enc_lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.encoder_hidden_units)

        # encoder_cell = tf.contrib.rnn.MultiRNNCell([enc_lstm_cell for _ in range(layer)], state_is_tuple=True)
        #encoder_cell = enc_lstm_cell

        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
            enc_lstm_cell, encoder_inputs_embedded,sequence_length=self.input_sequence_length,
            dtype=tf.float32,time_major=True
        )
        """attention"""
        attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
        # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            self.decoder_hidden_units, attention_states,memory_sequence_length=self.input_sequence_length)

        dec_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.decoder_hidden_units)
        # decoder_cell = tf.contrib.rnn.MultiRNNCell([dec_lstm_cell for _ in range(layer)], state_is_tuple=True)
        #decoder_cell = dec_lstm_cell
        # DONE: SET alignment_history=True if record attention are needed
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
            dec_lstm_cell, attention_mechanism,
            alignment_history=True,
            attention_layer_size=200)
        """project layer"""
        output_layer = layers_core.Dense( self.des_vocab_size , use_bias=False, name="output_projection")
        """attention wiht decoder"""
        # Helper
        if self.is_trainging == 'train' or self.is_trainging == 'test':
            print("In train of test set")
            #training mode
            maximum_iterations= None
            helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_inputs_embedded, self.decoder_sequence_length, time_major=True)
        elif self.is_trainging is 'inference' :
            #inference
            maximum_iterations  = tf.round(tf.reduce_max(self.input_sequence_length) *2)
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.decoder_embeddings,
                tf.fill([self.batch_size], self.tgt_sos_id), self.tgt_eos_id)
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            attention_cell, helper, attention_cell.zero_state(dtype=tf.float32,batch_size=self.batch_size),
            output_layer= output_layer )
        # Dynamic decoding
        outputs, final_context_state, self.final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations = maximum_iterations)
        # This means the pre.
        # Can also be used as  self.translations = outputs.sample_id
        # TODO:Delete the useless statment.
        # This is useless
        self.translations =tf.argmax(outputs.rnn_output,axis=2)

        decoder_logits = outputs.rnn_output
        decoder_logits_T = tf.transpose(decoder_logits,[1,0,2])
        self.decoder_prediction = tf.argmax(decoder_logits_T, 2)
        decoder_targets_T =  tf.transpose(self.decoder_targets,[1,0])



        """calcute the loss"""
        target_label = decoder_targets_T
        stepwise_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_label,
            logits=decoder_logits
        )
        # TODO: make a check whether it is time_major
        max_time = tf.shape(self.decoder_targets)[0]
        target_weights = tf.sequence_mask( self.target_sequence_length, max_time, dtype=decoder_logits.dtype)
        """caculate loss multy by sequence_mask"""
        self.loss = tf.reduce_mean(stepwise_cross_entropy *target_weights )/tf.to_float(self.batch_size)

        """check whether to inference"""
        if self.is_trainging != 'train':
            self.translations = outputs.sample_id  # the size of translations is [batch_size,sentence_length]
            return

        """apply clipped_gradients"""
        parameters = tf.trainable_variables()
        gradients = tf.gradients(self.loss, parameters)
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        """add train op"""
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, parameters))

        summary_loss = tf.summary.scalar("training_loss",self.loss)

        """add summary"""
        # atention summary
        attention_images = (final_context_state.alignment_history.stack())
        # Reshape to (batch, src_seq_len, tgt_seq_len,1)
        attention_images = tf.expand_dims(
          tf.transpose(attention_images, [1, 2, 0]), -1)
        # Scale to range [0, 255]
        attention_images *= 255
        attention_summary = tf.summary.image("attention_images", attention_images)
        # other variable
        tf.summary.scalar("Training_Loss", self.loss)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar("gradient_norm", gradient_norm)
        tf.summary.histogram("dec_lstm_cell.trainable_weights[0]",dec_lstm_cell.trainable_weights[0])


        self.summaries = tf.summary.merge_all()

def run_epoch(sess,reader,model,writer,global_step):
    result = reader.NextBatch('train') # modify the data set
    average_loss   = 0
    """modify this when change data set"""
    for i in range(reader.train_batch_length): # modify this when change data set
        idx,idy = result.__next__()
        fd =reader.next_feed(model,idx,idy)
        start = time.time()
        _, l = sess.run([model.train_op, model.loss], fd)
        average_loss+=l
        end  =time.time()
        if i == 0 :
            #end  =time.time()
            summa, predict_ , final_length,local_loss = sess.run([model.summaries,
                                                                model.decoder_prediction,
                                                                model.final_sequence_lengths,
                                                                model.loss], fd)
            writer.add_summary(summa)
            src = [ reader.id_to_word(item,"src")  for item in idx[1]]
            aim = [ reader.id_to_word(item,"des")  for item in idy[1]]
            pre = [ reader.id_to_word(item,"des")  for item in predict_.T[1]]
            print('  minibatch loss:{}'.format(l))
            print('  src: {}'.format(' '.join(src)))
            print('  aim: {}'.format(' '.join(aim)))
            print('  pre: {}'.format(' '.join(pre[:final_length[1]])))
            #start = time.time()
        """modify this when change data set"""
    return average_loss/(reader.train_batch_length)#
def run_test(sess,reader,model):
    result = reader.NextBatch('test')
    average_loss   = 0
    start = time.time()
    for i in range(reader.dev_batch_length ):
        idx,idy = result.__next__()
        fd =reader.next_feed(model,idx,idy)
        l = sess.run(model.loss, fd)
        average_loss+=l
    return average_loss/(reader.dev_batch_length)
def run_inference(sess,reader,model):
    result = reader.NextBatch('dev')
    idx,idy = result.__next__()
    fd =reader.next_feed(model,idx,idy)
    translations = sess.run([model.translations], fd)
    src = [ reader.id_to_word(item,'src')  for item in idx[0]]
    aim = [ reader.id_to_word(item,'des')  for item in idy[0]]
    pre = [ reader.id_to_word(item,'src')  for item in translations[0][0]]
    print('\nInference\n')
    print('  src: {}'.format(' '.join(src)))
    print('  aim: {}'.format(' '.join(aim)))
    print('  inf: {}'.format(' '.join(pre)))
