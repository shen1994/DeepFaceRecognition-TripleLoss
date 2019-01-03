# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:54:22 2018

@author: shen1994
"""

import os
import numpy as np
from facenet_utils import image_crop_and_resize
from facenet_utils import calculate_distance
from i_model import inception_v2
from keras import backend as K

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    '''
    # face vector model
    vector_graph_def = tf.GraphDef()
    vector_graph_def.ParseFromString(open("model/pico_FaceVector_model.pb", "rb").read())
    vector_tensors = tf.import_graph_def(vector_graph_def, name="")
    vector_sess = tf.Session()
    vector_opt = vector_sess.graph.get_operations()
    vector_x = vector_sess.graph.get_tensor_by_name("vector_input:0")
    vector_y = vector_sess.graph.get_tensor_by_name("lambda_1/l2_normalize:0") # embeddings/BiasAdd:0
    '''
    model = inception_v2((192, 192, 3))
    model.load_weights('model/weights.35.hdf5', by_name=True)
    # model.load_weights('C:/Users/shen1994/Desktop/DeepFaceVectorOnTripleLoss/model/weights.55.hdf5', by_name=True)
    vector_sess = K.get_session()
    vector_x = vector_sess.graph.get_tensor_by_name("vector_input:0")
    vector_y = vector_sess.graph.get_tensor_by_name("lambda_1/l2_normalize:0") # fc1/BiasAdd:0
    
    # part 1
    anchor_image = image_crop_and_resize("images/test1/test.jpg")
    anchor_vector = vector_sess.run(vector_y, feed_dict={vector_x: [anchor_image]})[0]

    pos_distances = []
    for one_pos in os.listdir("images/test1/pos"):
        positive_image = image_crop_and_resize("images/test1/pos" + os.sep + one_pos)
        positive_vector = vector_sess.run(vector_y, feed_dict={vector_x: [positive_image]})[0]
        positive_distance = calculate_distance(anchor_vector, positive_vector)
        pos_distances.append(positive_distance)
        
    neg_distances = []
    neg_names = []
    for one_neg in os.listdir("images/test1/neg"):
        neg_names.append(one_neg)
        negtive_image = image_crop_and_resize("images/test1/neg" + os.sep + one_neg)
        negtive_vector = vector_sess.run(vector_y, feed_dict={vector_x: [negtive_image]})[0]
        negtive_distance = calculate_distance(anchor_vector, negtive_vector)
        neg_distances.append(negtive_distance)
    
    half_score = (np.mean(np.array(pos_distances)) + np.mean(np.array(neg_distances))) / 2.0

    print(np.mean(np.array(pos_distances)), np.mean(np.array(neg_distances)), half_score)

    error = 0
    half_error = 0
    for pos in pos_distances:
        if pos > half_score:
            half_error += 1
            print(pos)
        if pos > 1.05:
            error += 1
            
    print('error: %d, half: %d' %(error, half_error))

    error = 0
    half_error = 0
    counter=0
    for neg in neg_distances:
        counter+=1
        if neg < half_score:
            half_error += 1
        if neg < 1.05:
            print(neg, neg_names[counter-1])
            error += 1
            
    print('error: %d, half: %d' %(error, half_error))

    # part 2
    anchor_image = image_crop_and_resize("images/test/test.jpg")
    anchor_vector = vector_sess.run(vector_y, feed_dict={vector_x: [anchor_image]})[0]

    pos_distances = []
    for one_pos in os.listdir("images/test/pos"):
        positive_image = image_crop_and_resize("images/test/pos" + os.sep + one_pos)
        positive_vector = vector_sess.run(vector_y, feed_dict={vector_x: [positive_image]})[0]
        positive_distance = calculate_distance(anchor_vector, positive_vector)
        pos_distances.append(positive_distance)
        
    neg_distances = []
    for one_neg in os.listdir("images/test/neg"):
        negtive_image = image_crop_and_resize("images/test/neg" + os.sep + one_neg)
        negtive_vector = vector_sess.run(vector_y, feed_dict={vector_x: [negtive_image]})[0]
        negtive_distance = calculate_distance(anchor_vector, negtive_vector)
        neg_distances.append(negtive_distance)
    
    half_score = (np.mean(np.array(pos_distances)) + np.mean(np.array(neg_distances))) / 2.0

    print(np.mean(np.array(pos_distances)), np.mean(np.array(neg_distances)), half_score)

    error = 0
    half_error = 0
    for pos in pos_distances:
        if pos > half_score:
            half_error += 1
        if pos > 1.05:
            error += 1
            
    print('error: %d, half: %d' %(error, half_error))

    error = 0
    half_error = 0
    counter=0
    for neg in neg_distances:
        counter+=1
        if neg < half_score:
            half_error += 1
        if neg < 1.05:
            error += 1
            
    print('error: %d, half: %d' %(error, half_error))
        
        
        
        
        
        
        
