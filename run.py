import configs
from esim import ESIM
import tensorflow as tf
import os
import pickle
from dataProcess import Preprocess
from batch import BatchGenerator
import numpy as np
import time
import copy

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def fill_placeholder(data, model,config):

  batch_x,batch_y,batch_label,batch_x_len,batch_y_len,\
  max_x_len,max_y_len,batch_x_mask,batch_y_mask = data.next_batch(config.batch_size)

  labels=np.array(list(map(int, batch_label)))
  label=convert_to_one_hot(labels, 2)

  feed_dict = {
                model.x:np.array(batch_x) ,
                model.y:np.array(batch_y),
                model.label:np.array(label),
                model.x_mask:np.array(batch_x_mask),
                model.y_mask:np.array(batch_y_mask),
                model.x_len :np.array(batch_x_len),
                model.y_len :np.array(batch_y_len),
                model.max_s1_len:max_x_len,
                model.max_s2_len: max_y_len,
                model.learning_rate:config.learning_rate
                }
  return feed_dict


def get_embedding(embedding_dir,embedding_file):
    file_path=os.path.join(embedding_dir,embedding_file)
    if not os.path.exists(file_path):
        print('%s is not exist!' % file_path)
    else:
        with open(file_path,'rb') as fe:
            embedding=pickle.load(fe)

        return np.array(embedding)

def getbatchdata(data):

    # FLAGS.vocab_size = data.word_size
    train_sentence1 = data.get_s1()
    train_sentence2 = data.get_s2()
    train_label = data.get_labels()
    train_sen1_length = data.get_s1_length()
    train_sen2_length = data.get_s2_length()
    word_dict = data.get_word_dict()
    # FLAGS.train_size = len(train_label)

    databatch = BatchGenerator(train_sentence1, train_sentence2, train_label, train_sen1_length, train_sen2_length,
                                word_dict,True)
    return databatch


def run_epoch(session, data, model, config,global_step=0, optim=None, training=False,writerCurve=None):

    print('---check para:\tis_train:{},batchsize:{},learning_rate:{},drop_keep_prob:{}---'
          .format(config.is_train,config.batch_size,config.learning_rate,config.drop_keep_prob))

    losses = 0.0
    acc_total = 0.0

    true_label_total = []
    pred_label_total = []

    fetches = {
        "acc": model.accuracy,
        "loss": model.loss,
        "pred": model.pred,
        "label": model.label,

    }
    if optim is not None:
        fetches["optimer"] = optim

    epoch_size = int(data.get_size()/config.batch_size)
    for step in range(epoch_size):
        feed_dict = fill_placeholder(data, model, config)

        vals = session.run(fetches, feed_dict)
        acc = vals["acc"]
        loss = vals["loss"]

        losses += loss
        global_step += 1
        acc_total += acc
        pred_label_total.extend(np.argmax(vals["pred"], 1))
        true_label_total.extend(np.argmax(vals["label"], 1))
        if training and step % 200 == 0:
            print('global_step: %s train_acc: %s  batch_train_loss: %s' % (global_step, acc, loss))

    acc_average = acc_total / epoch_size
    loss_average = losses / epoch_size

    return acc_average, loss_average, global_step, pred_label_total, true_label_total


def train(config,data_train,data_dev):

    train_config=copy.deepcopy(config)
    train_config.is_train=True

    esim_model = ESIM(config)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    start_time = time.time()
    best_accuracy = 0.0
    best_val_epoch = 0
    last_change_epoch = 0
    decay_rate = 1
    global_step=0


    ckpt_file = config.restore_path
    ckpt = tf.train.get_checkpoint_state(ckpt_file)
    if ckpt and ckpt.model_checkpoint_path:
        print("restore model from: %s " % (ckpt.model_checkpoint_path))
        saver.saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(config.MAXITER):
        train_acc, train_loss, train_global_step, train_pred_total, train_true_total = run_epoch(
                                                                                            sess,
                                                                                            data=data_train,
                                                                                            model=esim_model,
                                                                                            config=train_config,
                                                                                            global_step=global_step,
                                                                                            optim=esim_model.optim,
                                                                                            training=True,
                                                                                           )
        print("Epoch: %d train_acc: %.3f train_loss %.3f train_global_step:%s"
              % (i, train_acc, train_loss, train_global_step))
        print('===============================================================')
        global_step=train_global_step

        dev_acc,dev_loss=evaluation(sess,i,data_dev,esim_model,config)

        if i > 1 and dev_acc > best_accuracy:
            saver.save(sess, config.save_path + 'trainmodel', global_step=i)
            best_accuracy = dev_acc
            best_val_epoch = i
            print('保存%d轮模型，准确率为：%f.3' % (i, best_accuracy))

        if (i - best_val_epoch > config.max_epoch_changelr) and (i-last_change_epoch)> config.changelr_interval:

            if config.learning_rate > config.min_lr:
                lr_decay = config.lr_decay **decay_rate
                new_learning_rate = config.learning_rate * lr_decay
                decay_rate += 1
                last_change_epoch=i
                print("learning_rate-->change!Dang!Dang!Dang!-->%.10f" % (new_learning_rate))
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        end_time = time.time()
        print(
            "--------  one_epoch time: %s\n " % ((end_time - start_time) // 60))
        if i - best_val_epoch > config.early_stopping:
            print("best_val_epoch:%d  best_val_accuracy:%.3f" % (best_val_epoch, best_accuracy))
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            break
        elif i == config.MAXITER - 1:
            print("best_val_epoch:%d  best_val_accuracy:%.3f" % (best_val_epoch, best_accuracy))
            print("Finishe Training")

    end_time = time.time()
    print(
        "-------- all_training time: %s \n " %  ((end_time - start_time) // 60))

def  evaluation(sess,epoch,data_dev,model,config):
    eval_config=copy.deepcopy(config)
    eval_config.batch_size=1
    dev_acc, dev_loss, _,  dev_pred_total, dev_true_total = \
        run_epoch(sess,data=data_dev,model=model,config=eval_config)
    print("Epoch: %d dev_acc: %.3f dev_loss %.3f" % (epoch, dev_acc, dev_loss))
    print('===============================================================')
    return dev_acc,dev_loss

def test(data_test,config):
    test_config = copy.deepcopy(config)
    test_config.batch_size = 1
    esim_test = ESIM(test_config)

    with tf.Session() as sess:

        saver=tf.train.Saver()
        ckpt_file = config.save_path
        ckpt = tf.train.get_checkpoint_state(ckpt_file)
        if ckpt and ckpt.model_checkpoint_path:
            print("restore model from: %s " % (ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

        test_acc, test_loss, _, test_pred_label, test_true_label =run_epoch(sess,data_test,esim_test,test_config)

        print("\n\n=================testing=======================")
        print("test_acc: %.3f test_loss %.3f" % (test_acc, test_loss))
        print('pred:\n',test_pred_label)
        print('true:\n', test_true_label)

def main():
    config = configs.parser_args()

    if os.path.exists(os.path.join(config.embedding_dir,config.embedding_file)):
        pretrain_embedding = get_embedding(config.embedding_dir, config.embedding_file)
        config.word_embedding=pretrain_embedding

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    if not os.path.exists(config.restore_path):
        os.makedirs(config.restore_path)
    if not os.path.exists(config.summary):
        os.makedirs(config.summary)


    if config.train:
        traindata = Preprocess(config.data_dir, config.train_file, config.word_dict_file)
        data_train = getbatchdata(traindata)

        devdata = Preprocess(config.data_dir, config.dev_file, config.word_dict_file)
        data_dev = getbatchdata(devdata)

        vocab_size = traindata.get_vocab_size()
        config.vocab_size = vocab_size

        train(config, data_train, data_dev)

    if config.predict:
        testdata = Preprocess(config.data_dir, config.test_file, config.word_dict_file)
        data_test = getbatchdata(testdata)
        test(data_test, config)



if __name__=="__main__":
    main()
