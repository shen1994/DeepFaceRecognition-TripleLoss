# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:11:17 2018

@author: shen1994
"""

import os
import cv2
import numpy as np
from utils import get_dataset
from utils import sample_people
from utils import select_triplets

class Generator(object):
    
    # people_per_batch > images_per_person: 偏向负样本，并且loss不会下降
    # people_per_batch < images_per_person: 偏向正样本，相当于每次从总样本抽取一小批次数据
    # images_per_person >= 40
    
    def __init__(self, model, input_op, output_op,
             path=None,
             class_number=19,
             people_per_batch=10, #35
             images_per_person=40, #5
             step_of_epoch=3000,
             alpha=0.2, 
             batch_size=8,
             image_shape=(220, 220, 3), 
             is_enhance=False):
        self.path = path
        self.model = model
        self.input_op = input_op
        self.output_op = output_op
        self.class_number = class_number
        self.people_per_batch = people_per_batch
        self.images_per_person = images_per_person
        self.step_of_epoch = step_of_epoch
        self.alpha = alpha
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.is_enhance = is_enhance
        
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])
        
    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * 0.5
        alpha += 1 - 0.5
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * 0.5 
        alpha += 1 - 0.5
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * 0.5
        alpha += 1 - 0.5
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * 0.5
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
  
    def horizontal_flip(self, img):
        rand = np.random.random()
        if  rand < 0.5:
            img = img[:, ::-1]
        return img
        
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
            if self.is_enhance:
                witch_one = np.random.randint(3)
                if witch_one == 0:
                    img = self.saturation(img)
                elif witch_one == 1:
                    img = self.brightness(img)
                else:
                    img = self.contrast(img)
                img = self.lighting(img)
                img = self.horizontal_flip(img)
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
        
            # generate enbeddings from parts
            image_length = len(image_paths)
            epochs = image_length // 64
            other = image_length % 64
            if epochs == 0:
                embeddings = self.model.run(self.output_op, \
                                            feed_dict={self.input_op: np.stack(self.get_images(image_paths))})
            else:
                embeddings = self.model.run(self.output_op, \
                                            feed_dict={self.input_op: np.stack(self.get_images(image_paths[:64]))})
                for epoch in range(epochs):
                    if epoch == 0:
                        continue
                    one = self.model.run(self.output_op, \
                                         feed_dict={self.input_op: np.stack(self.get_images(image_paths[epoch*64:(epoch+1)*64]))})
                    embeddings = np.concatenate((embeddings, one), axis=0)
                start = epochs * 64
                if not other == 0:
                    one = self.model.run(self.output_op, \
                                         feed_dict={self.input_op: np.stack(self.get_images(image_paths[start:start+other]))})
                    embeddings = np.concatenate((embeddings, one), axis=0)
            
            # select hard-distinguish triplets
            triplets, labels, length = select_triplets(embeddings, num_per_class, 
                                                       image_paths, self.people_per_batch, alpha=self.alpha)
            
            counter = 0
            batch_counter = 0
            images_pixels = []
            images_labels = []
    
            print("\nGenerate " + str(length) + " Once Again!")
            
            max_feeds = self.step_of_epoch * self.batch_size
            if length > max_feeds:
                length = max_feeds
            
            for i in range(length):
                if (batch_counter + 1) * self.batch_size > length:
                    counter = 0
                    batch_counter = 0
                    images_pixels = []
                    images_labels = []
                    break
                images_pixels.extend(np.stack(self.get_images(triplets[i])))
                images_labels.extend(np.stack(labels[i]))
                counter += 1
                if counter == self.batch_size:
                    yield (np.array(images_pixels), np.array(images_labels))
                    counter = 0
                    batch_counter +=  1
                    images_pixels = []
                    images_labels = []  
    
      