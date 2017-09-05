import data
import attention_model as model
import tensorflow as tf
import os
import time

batch_size = 20
max_time_step = 150 # also is the max length of the sentence
input_embedding_size =100
encoder_hidden_units= 100
attention_unites = 100
decoder_hidden_units = encoder_hidden_units
summary_path = 'translate_atten'
mode_restore_path = 'translate_atten/model/model.ckpt'

"""parameter"""
learning_rate= 0.0001
layer = 2

train_reader =data.PTBreader(
                  src_train_path = "./nmt_data/train.vi",
                  des_train_path = "./nmt_data/train.en",
                  src_test_path  = "./nmt_data/tst2012.vi",
                  des_test_path  = "./nmt_data/tst2012.en",
                  src_dev_path   = "./nmt_data/tst2013.vi",
                  des_dev_path   = "./nmt_data/tst2013.en",
                  src_vocab_path  = "./nmt_data/vocab.vi",
                  des_vocab_path  = "./nmt_data/vocab.en",
                  batch_size =batch_size,
                  max_time_step=max_time_step
                 )

with tf.variable_scope("language_model",reuse=None):
    train_model = model.Seq2seq(
        max_time_step =max_time_step,
        batch_size= batch_size,
        encoder_hidden_units =encoder_hidden_units ,
        src_vocab = train_reader.source_voca,
        des_vocab = train_reader.target_voca,
        input_embedding_size = input_embedding_size,
        decoder_hidden_units = decoder_hidden_units,
        learning_rate = learning_rate,
        is_trainging  ='train',
        layer = layer,
        tgt_sos_id = train_reader.START_ID,
        tgt_eos_id = train_reader.EOS_ID
        )
    train_model.create_train_model()
with tf.variable_scope("language_model",reuse=True):
    eval_model = model.Seq2seq(
        max_time_step =max_time_step,
        batch_size= batch_size,
        encoder_hidden_units =encoder_hidden_units ,
        src_vocab = train_reader.source_voca,
        des_vocab = train_reader.target_voca,
        input_embedding_size = input_embedding_size,
        decoder_hidden_units = decoder_hidden_units,
        learning_rate = learning_rate,
        is_trainging  ='test',
        layer = layer,
        tgt_sos_id = train_reader.START_ID,
        tgt_eos_id = train_reader.EOS_ID
    )
    eval_model.create_train_model()


sess= tf.Session()
saver = tf.train.Saver()
"""restore"""
try:
    print("restore model...")
    #saver.restore(sess, "save_path/file_name.ckpt-???") 
    saver.restore(sess,mode_restore_path)
except:
    print("canot find model ,now start initializer variables")
    sess.run(tf.global_variables_initializer())
"""save graph"""
if os.path.exists(summary_path) is True:
    graph=  None
else:
    os.mkdir(summary_path)
    graph=  tf.get_default_graph()
    print('save graph to {}'.format(summary_path))
writer = tf.summary.FileWriter(summary_path,graph=graph)

for epoch in range(500):
    start = time.time()
    avg_train_loss= model.run_epoch(sess,train_reader,train_model,writer,global_step=epoch)
    _,rate = sess.run([train_model.add_global,train_model.learning_rate])
    end = time.time()
    print('Epoch:{} used time :{} avg_train_loss is :{}'.format(
                                                    epoch,end-start,avg_train_loss))
    if epoch %5 ==0 :
        print('learning_rate is rate:{}'.format(rate))
        print('save model to > {}'.format(mode_restore_path))
        saver.save(sess,mode_restore_path,global_step = epoch)
        start = time.time()
        avg_test_loss = model.run_test(sess,train_reader,eval_model)
        model.run_inference(sess,train_reader,eval_model )
        end = time.time()
        print('Test  used time :{} avg_test_loss is :{}'.format(
                                                       end-start,avg_test_loss))
