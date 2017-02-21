import numpy as np
import pandas as pd
import pickle
import cv2
import os
import random
import matplotlib.image as mpimg
import glob

    
def extract_images_from_dir(path, comb=False):
    '''extract images from a specified directory
    
    Notes: 
        if comb==True, only Recombination images will be extracted
           otherwise, all iamges will be extracted    
    Args:
        path (string): a path to the directory of images
        comb (boolean):
                True: use reconstruct images
                False: use all images
    Return:
        filtered_images (list): a python list containing a list of filtered images
    '''
    # filtered_images = path + '/.jpg'
    image_all = glob.glob(os.path.join(path, '*.jpg'))
    print(image_all)
    if comb:
        filtered_images = [image for image in image_all if 'Recombination' in image]
    else:
        filtered_images = image_all
        
    return filtered_images


## for online purpose
## loading query image

from dqlib.imutils import rgb2gray

def extract_normal_features(images):
    '''
    '''
    random_image = random.choice(images)
    image_loaded = mpimg.imread(random_image)
    dataset = np.ndarray(shape=(len(images),
                         image_loaded.shape[0],
                         image_loaded.shape[1]),
                         dtype=np.float32)
    

    for index, image in enumerate(images):

        rgb_image = mpimg.imread(image)
        print(rgb_image.shape)
        
        gray_image = rgb2gray(rgb_image)
        nor_image = (gray_image.astype(float) - 255/2) / 255
        
        
        dataset[index,:,:] = nor_image
        
    return dataset




def load_queries(path):
    '''
    '''
    dataset_all = []
    for angle_dir in [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]:
        print(angle_dir)
        images = extract_images_from_dir(os.path.join(path, angle_dir), comb=False)
        # print(images)
        # normalize the image
        dataset = extract_normal_features(images)
        # 
        #dataset_all.append(dataset.reshape(dataset.shape[0], dataset.shape[1]*dataset.shape[2]))
        dataset_all.append(dataset)
        
    dataset_final = np.concatenate(dataset_all)
    
    return dataset_final