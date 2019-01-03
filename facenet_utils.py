# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:15:27 2018

@author: shen1994
"""

import cv2
import numpy as np
from PIL import Image

def prepare_whiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def image_crop_and_resize(path):
    
    # img = Image.open(path)
    # img = img.resize((192, 192))
    # img = img.crop([11, 11, 310, 310]) # 299 * 299
    img = cv2.imread(path)
    img = cv2.resize(img, (192, 192))
    # x = np.array(img)
    x = prepare_whiten(img)
    
    return x

def calculate_distance(vec1, vec2):

    vec_dot = np.dot((vec1-vec2), (vec1-vec2).T)
    
    return np.sqrt(vec_dot)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
