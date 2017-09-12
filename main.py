import data
import attention_model as model
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import argparse
import evalution_utils
"""parameter"""
def add_arguments(parser):
    parser.add_argument("--src_language",type=str,default="cs",help="the source of language")
    parser.add_argument("--des_language",type=str,default="en",help="the target of language")
    parser.add_argument("--summary_path",type=str,default="translate_atten",help="the path of summary")


nmt_parser = argparse.ArgumentParser()
add_arguments(nmt_parser)
args = nmt_parser.parse_args()

#TODO: convert these variables to parameters
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # close warning
batch_size = 20
max_time_step = 150 # also is the max length of the sentence
input_embedding_size =100
encoder_hidden_units= 100
attention_unites = 100
decoder_hidden_units = encoder_hidden_units
summary_path = args.summary_path
#mode_restore_path = os.path.join(args.summary_path,"model/model.ckpt")
mode_restore_path = 'translate_atten/model'

learning_rate= 0.0001
layer = 2
src_language ='.'+ args.src_language
des_language ='.'+ args.des_language


train_reader =data.PTBreader(
                  src_train_path = "./nmt_data/train"+src_language,
                  des_train_path = "./nmt_data/train"+des_language,
                  src_test_path  = "./nmt_data/tst2012"+src_language,
                  des_test_path  = "./nmt_data/tst2012"+des_language,
                  src_dev_path   = "./nmt_data/tst2013"+src_language,
                  des_dev_path   = "./nmt_data/tst2013"+des_language,
                  src_vocab_path  = "./nmt_data/vocab"+src_language,
                  des_vocab_path  = "./nmt_data/vocab"+des_language,
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
  ckpt_state = tf.train.get_checkpoint_state(mode_restore_path)
except tf.errors.OutOfRangeError as e:
  tf.logging.error('Cannot restore checkpoint: %s', e)

if ckpt_state != None :
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)
else:
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

train_loss = []
test_loss  = []

# use sess.as_default() we can directly use  model.global_step.eval() instead of sess.run()
# may sess.as_default() only need to be used once ? I will checkout it
with sess.as_default():
    for epoch in range(500):
        start = time.time()
        # get  avg_train_loss
        avg_train_loss= model.run_epoch(sess,train_reader,train_model,writer)
        train_loss.append(avg_train_loss)
        rate = train_model.learning_rate.eval()
        end = time.time()
        print('Epoch:{} used time :{} avg_train_loss is :{}'.format(
                                                        epoch,end-start,avg_train_loss))
        if epoch %5 ==0 :
            print('learning_rate is rate:{}'.format(rate))
            print('save model to > {}'.format(mode_restore_path))
            saver.save(sess,mode_restore_path,global_step = train_model.global_step.eval())
            start = time.time()
            # get test_loss
            avg_test_loss = model.run_test(sess,train_reader,eval_model)
            test_loss.append(avg_test_loss)
            end = time.time()
            print('Test  used time :{} avg_test_loss is :{}'.format(
                                                           end-start,avg_test_loss))
        if epoch %10 ==0:
           print("strart inference:{}".format(epoch))
           model.run_inference(sess,train_reader,eval_model )
           #calcute the bleu
           score = evalution_utils.evaluate("nmt_data/tst2013."+ des_language , "nmt_output", "bleu", bpe_delimiter=None)
           print("epoch:{} , bleu:{}".format(epoch,score))

# save figure
plt.plot(test_loss)
plt.savefig('test_loss.png')
plt.plot(train_loss)
plt.savefig('trainloss.png')
