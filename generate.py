# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:11:17 2018

@author: shen1994
"""

import os
import cv2
import time
import numpy as np
import tensorflow as tf
from keras import backend as K
from utils import get_dataset
from utils import sample_people
from utils import select_triplets
from i_model import inception_v2

class Generator(object):
    
    # people_per_batch > images_per_person: 偏向负样本，并且loss不会下降, wrong
    # people_per_batch < images_per_person: 偏向正样本，相当于每次从总样本抽取一小批次数据, corect
    # images_per_person >= 3
    
    def __init__(self,
             path,
             class_number=19,
             people_per_batch=16, # 10->8->16->16
             images_per_person=10, # 40->40->20->10
             alpha=0.2,
             step_of_epoch=2000,
             batch_size=16, # batch_size = people_per_batch * n, n=1,2,3...
             image_shape=(192, 192, 3),
             embedding_size=512):
        # init all varibles
        self.path = path
        self.class_number = class_number
        self.people_per_batch = people_per_batch
        self.images_per_person = images_per_person
        self.alpha = alpha
        self.step_of_epoch = step_of_epoch
        self.batch_size = batch_size
        self.max_feeds = self.step_of_epoch * self.batch_size
        self.image_shape = image_shape
        self.embedding_size = embedding_size
        
        # get the model from center-loss and redefine a graph for data feed
        vector_graph_def = tf.GraphDef()
        vector_graph_def.ParseFromString(open("model/pico_FaceVector_model.pb", "rb").read())
        vector_graph = tf.import_graph_def(vector_graph_def, name="")
        self.model = tf.Session(graph=vector_graph)
        self.model.graph.get_operations()
        self.input_op = self.model.graph.get_tensor_by_name("feed_input:0")
        self.output_op = self.model.graph.get_tensor_by_name("feed_output/l2_normalize:0")
        
    def prepare_whiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def get_images(self, paths):
        images = []
        for path in paths:
            img = cv2.imread(path)
            img = cv2.resize(img, (self.image_shape[0], self.image_shape[1]))
            x = self.prepare_whiten(img)
            images.append(x)
        return images
        
    def get_path_prefix(self, path):
        prefix = ''
        counter = 0
        for index in range(len(path)):
            if (path[index] == '/' or path[index] == '\\'):
                counter += 1
            if (path[index] == '/' or path[index] == '\\') and counter == 3:
                break
            prefix += path[index]
        return prefix
        
    def generate(self):
        
        while(True):
            
            samples_time = 0
            samples_ntime = 0
            samples_counter = 0
            samples_ncounter = 0
            g_samples_counter = 0
            samples_triplets, samples_labels = [], []
            
            # it is neccessary to use the same model to generate enough samples!
            while(samples_counter < self.max_feeds):
                
                # calculate current time
                time_start = time.time()
                
                # random select samples
                dataset = get_dataset(self.path, people_number=self.people_per_batch)
                image_paths, num_per_class = sample_people(dataset, self.people_per_batch, self.images_per_person)
    
                # random select more negtive samples
                
                neg_samples = self.images_per_person * self.people_per_batch
                neg_counter = 0
                class_list = [self.path + os.sep + one for one in os.listdir(self.path)]
                class_index = 0
                for one_num in num_per_class:
                    class_list.remove(self.get_path_prefix(image_paths[class_index]))
                    class_index += one_num
                while(neg_counter < neg_samples):
                    rand_index = np.random.randint(len(class_list))
                    name = np.random.choice(os.listdir(class_list[rand_index]))              
                    image_paths.append(class_list[rand_index] + os.sep + name)
                    neg_counter += 1               
    
                # generate enbeddings from the part of samples
                min_cell = 64
                image_length = len(image_paths)
                epochs = image_length // min_cell
                other = image_length % min_cell
                if epochs == 0:
                    embeddings = self.model.run(self.output_op, \
                                                feed_dict={self.input_op: np.stack(self.get_images(image_paths))})
                else:
                    embeddings = self.model.run(self.output_op, \
                                                feed_dict={self.input_op: np.stack(self.get_images(image_paths[:min_cell]))})
    
                    for epoch in range(epochs):
                        if epoch == 0:
                            continue
                        one = self.model.run(self.output_op, \
                                             feed_dict={self.input_op: np.stack(self.get_images(image_paths[epoch*min_cell:(epoch+1)*min_cell]))})
                        embeddings = np.concatenate((embeddings, one), axis=0)
                    start = epochs * min_cell
                    if not other == 0:
                        one = self.model.run(self.output_op, \
                                             feed_dict={self.input_op: np.stack(self.get_images(image_paths[start:start+other]))})
                        embeddings = np.concatenate((embeddings, one), axis=0)
                
                # select hard-distinguish triplets
                triplets, labels, length = select_triplets(embeddings, num_per_class, 
                                                           image_paths, self.people_per_batch, alpha=self.alpha)
                
                # add to total samples
                samples_triplets.extend(triplets)
                samples_labels.extend(labels)
                samples_counter += length
                samples_ncounter += length
                g_samples_counter += 1
                
                samples_temp_time = time.time() - time_start
                samples_ntime += samples_temp_time
                samples_time += samples_temp_time
                
                if g_samples_counter % 10 == 0:
                    print("\nG: " + str(g_samples_counter) + "---Generate Cell: " + str(samples_ncounter) +  
                          "---Generate: " + str(samples_counter) + "---Time: " + str(samples_ntime) + " ......")
                    samples_ntime = 0
                    samples_ncounter = 0
            
            # samples random shuffle
            samples_random = [i for i in range(samples_counter)]
            np.random.shuffle(samples_random)
            samples_triplets = [samples_triplets[one] for one in samples_random]
            samples_labels = [samples_labels[one] for one in samples_random]

            # not extend the maximum
            if samples_counter > self.max_feeds:
                samples_counter = self.max_feeds
    
            print("\nGenerate: " + str(samples_counter) + "---Time: " + str(samples_time) + "---Once Again!")

            # feed data using all selected samples
            counter = 0
            images_pixels = []
            images_labels = []            
            for i in range(samples_counter):
                images_pixels.extend(np.stack(self.get_images(samples_triplets[i])))
                images_labels.extend(np.stack(samples_labels[i]))
                counter += 1
                if counter == self.batch_size:
                    yield np.array(images_pixels), np.array(images_labels)
                    counter = 0
                    images_pixels = []
                    images_labels = []  
    
    
    
    
    
    