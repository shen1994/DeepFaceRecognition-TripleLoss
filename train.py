# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:08:13 2018

@author: shen1994
"""

import os
import keras
from keras import backend as K
from i_model import inception_v2
from generate import Generator

def triplet_loss(y_true, y_pred, alpha=0.2, embedding_size=512):
    
    r_pred = K.reshape(y_pred, [batch_size, 3, embedding_size])
    size = int(r_pred.get_shape()[0])
    anchor = K.reshape(r_pred[0, 0, :], [1, embedding_size])
    positive = K.reshape(r_pred[0, 1, :], [1, embedding_size])
    negtive = K.reshape(r_pred[0, 2, :], [1, embedding_size])
    for i in range(1, size):
        anchor = K.concatenate((anchor, K.reshape(r_pred[i, 0, :], [1, embedding_size])), axis=0)
        positive = K.concatenate((positive, K.reshape(r_pred[i, 1, :], [1, embedding_size])), axis=0)
        negtive = K.concatenate((negtive, K.reshape(r_pred[i, 2, :], [1, embedding_size])), axis=0)
        
    d_pos = K.sum(K.square(anchor - positive), axis=-1)
    d_neg = K.sum(K.square(anchor - negtive), axis=-1) # not neg, because neg is random, but pos is all
    
    # N = 0.5 # N: not extend 0.5
    # threshold = 1.1 # threshold: get it from center-loss training
    # n_loss = K.mean(K.maximum(d_pos - d_neg + alpha, 0))
    # n_loss = tf.Print(n_loss, [n_loss], "n_lo")
    # p_loss = K.mean(K.maximum(K.sqrt(d_pos) - threshold, 0))
    # p_loss = tf.Print(p_loss, [p_loss], "p_lo")
    # loss = N * n_loss + p_loss
    
    loss = K.sum(K.maximum(d_pos - d_neg + alpha, 0))
    
    return loss 
    
    '''
    r_pred = tf.reshape(y_pred, [batch_size, 3, embedding_size])
    anchor, positive, negative = tf.unstack(r_pred, 3, 1)
    print(anchor, positive, negative)
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    basic_loss = tf.Print(basic_loss, [basic_loss], "bac")
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([loss] + regularization_losses, name='total_loss')
    return total_loss
    '''
    

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    model_path = 'model'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    epochs = 2000
    steps_per_epoch = 500 # 500 * 16 = 8000 samples
    batch_size = 16
    alpha = 0.2 # it should be the same as alpha in triple-loss
    embedding_size = 512
    image_shape = (192, 192, 3) # 96*96*3--->192*192*3--->224*224*3--->299*299*3?
     
    network = inception_v2(image_shape, embedding_size=embedding_size)
    network.load_weights('model/weights.70.hdf5', by_name=True)
    
    # 3e-4 ---> 3e-5
    opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True, decay=1e-6)
    network.compile(loss=triplet_loss, optimizer=opt)
    
    # forward network calculation，select hard-positive and hard-negtive
    # for those hard-positive, we take all positive pairs
    # for those hard-negtive, we take pairs randomly
    triplets_train_generate = Generator(path="images/train_align",
                                        class_number=len(os.listdir("images/train_align")),
                                        batch_size=batch_size,
                                        image_shape=image_shape, 
                                        embedding_size=embedding_size,
                                        step_of_epoch=steps_per_epoch,
                                        alpha=alpha)

    # backward network calculation，triplet calcution       
    callbacks = [keras.callbacks.ModelCheckpoint('model/weights.{epoch:02d}.hdf5',
                                                  verbose=1,
                                                  save_weights_only=True)]
    history = network.fit_generator(generator=triplets_train_generate.generate(), 
                                    epochs=epochs,
                                    steps_per_epoch=steps_per_epoch,
                                    verbose=1,
                                    initial_epoch=0,
                                    callbacks=callbacks,
                                    workers=1)         
  