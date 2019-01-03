# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:08:13 2018

@author: shen1994
"""

import os
import keras
import tensorflow as tf
from keras import backend as K
from cnn_model import facenet_cnn
from cnn_generate import Generator

def triplet_loss(y_true, y_pred, alpha=0.2, embedding_size=128):
    
    anchor, positive, negative = tf.unstack(tf.reshape(y_pred, [-1, 3, embedding_size]), 3, 1)
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=1)
    pos_dist = tf.log(tf.add(pos_dist, 1.0))
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=1)
    neg_dist = tf.log(tf.add(neg_dist, 1.0))
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # rate = tf.reduce_mean(pos_dist) / tf.reduce_mean(neg_dist)
    # loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0) # * rate
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0) # + 0.1 * tf.reduce_mean(pos_dist) 
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([loss] + regularization_losses, name='total_loss')
    return total_loss  

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    model_path = 'model'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    epochs = 1000
    steps_per_epoch = 200
    batch_size = 32
    embedding_size = 128
    alpha = 0.2
    image_shape = (224, 224, 3)
     
    network = facenet_cnn(image_shape, embedding_size=embedding_size)
    # network.summary()
    network.load_weights('model/weights.02.hdf5', by_name=True)
    opt = keras.optimizers.Adam(lr=1e-3, beta_1=0.99, beta_2=0.999, epsilon=0.1) # 0.05->0.001
    network.compile(loss=triplet_loss, optimizer=opt, metrics=['accuracy'])

    # forward network calculation，select hard-positive and hard-negtive
    # for those hard-positive, we take all positive pairs
    # for those hard-negtive, we take pairs randomly
    sess = K.get_session()
    network_input = sess.graph.get_tensor_by_name("vector_input:0")
    network_output = sess.graph.get_tensor_by_name("lambda_3/l2_normalize:0")
    triplets_generate = Generator(sess, 
                                  network_input, network_output, 
                                  path="images/train_align", 
                                  class_number=len(os.listdir("images/train_align")), 
                                  batch_size=batch_size,
                                  image_shape=image_shape, 
                                  step_of_epoch=steps_per_epoch,
                                  alpha=alpha, 
                                  is_enhance=True)

    # backward network calculation，triplet calcution       
    callbacks = [keras.callbacks.ModelCheckpoint('model/weights.{epoch:02d}.hdf5',
                                                  verbose=1,
                                                  save_weights_only=True)]
    history = network.fit_generator(generator=triplets_generate.generate(), 
                                    epochs=epochs,
                                    steps_per_epoch=steps_per_epoch,
                                    verbose=1,
                                    initial_epoch=2,
                                    callbacks=callbacks,
                                    workers=1)     
