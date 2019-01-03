# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:53:43 2018

@author: shen1994
"""

import os
import numpy as np
from six.moves import xrange
    
class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
        
def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths
  
def get_dataset(path, people_number=5):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]

    class_indexes = []
    class_number = len(classes)
    for i in range(people_number):
        random_index = np.random.randint(class_number)
        while random_index in class_indexes:
            random_index = np.random.randint(class_number)
        class_indexes.append(random_index)
    classes = [classes[index] for index in class_indexes]

    for i in range(people_number):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset
    
def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person
  
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    
    i= 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1
        
    return image_paths, num_per_class
        
def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha=0.2):
    """ Select the triplets for training
    """
    emb_start_idx = 0
    triplets = []
    labels = []
    
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in xrange(people_per_batch):
        one_triplets, one_labels = [], []
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
             # For every possible positive pair.
            for pair in xrange(j, nrof_images):
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))

                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN # delete same pairs  
                # second step, set the upper threshold to make the distances stable
                # semi_neg = np.where(np.logical_and(neg_dists_sqr < pos_dist_sqr + alpha, 
                #                                    pos_dist_sqr < neg_dists_sqr))[0] # facenet selection + samples number is too small
                semi_neg = np.where(neg_dists_sqr < pos_dist_sqr + alpha)[0] # VGG Face selecction # samples number is larger
                
                if len(semi_neg) > 0:
                    # all pos pairs and random negs
                    hard_idx = np.random.randint(len(semi_neg))
                    
                    # hard_idx = np.argmin([neg_dists_sqr[index] for index in semi_neg])
                    
                    n_idx = semi_neg[hard_idx]

                    one_triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    one_labels.append((embeddings[a_idx], embeddings[p_idx], embeddings[n_idx]))
                    
        triplets.append(one_triplets)
        labels.append(one_labels)

        emb_start_idx += nrof_images
        
    # confirm each batch, the number of different peoples is the same
    min_len = len(triplets[0])
    for one in triplets:
        one_len = len(one)
        if min_len > one_len:
            min_len = one_len
            
    new_triplets,  new_labels = [], []
    for each in range(len(triplets)):
        new_triplets.extend(triplets[each][:min_len])
        new_labels.extend(labels[each][:min_len])
     
    # random shuffle
    new_length = min_len * len(triplets)
    random_index = [i for i in range(new_length)]
    np.random.shuffle(random_index)
    new_triplets = [new_triplets[one] for one in random_index]
    new_labels = [new_labels[one] for one in random_index]
            
    return new_triplets, new_labels, new_length
