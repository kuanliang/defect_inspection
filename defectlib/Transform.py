import os
import numpy as np
import pickle
import cv2
from imutils import rotate

import Config

def make_arrays(nb_rows, angle):
    '''initialize empty numpy arrays for dataset and labels 
    
    
    
    '''
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_height[angle], img_width[angle]), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def align_tensor(tensor, angle, width=255):
    '''align and resize the tensor to univeral (255*255) scale
    
    Notes: 
    
    Args:
        tensor: tensor
        angle: angle
    
    Return:
        return a tensor with resized and angle adjusted angle
    
    '''
    # initialize an empty numpy array for storing image arrays
    empty_tensor = np.ndarray((tensor.shape[0], width, width))
    
    for index, image in enumerate(tensor):
        resized_image = cv2.resize(image, (width, width))
        rotated_image = rotate(resized_image, angle)
        
        empty_tensor[index,:,:] = rotated_image
        
    return empty_tensor 


def merge_datasets(pickle_folder):
    '''
    
    '''
    tensorList = []
    labelList = []
    for pickleFile in os.listdir(pickle_folder):
        # print label
        # print pickleFile
        if pickleFile.endswith('pickle'):
            print pickleFile
            label = int(os.path.splitext(pickleFile)[0].split('_')[-1].replace('c', ''))
            print label
            angle = os.path.splitext(pickleFile)[0].split('_')[1].replace('a', '')
            print angle
            try:
                with open(os.path.join(pickle_folder, pickleFile), 'rb') as f:
                    tensor = pickle.load(f)
                    tensorList.append(align_tensor(tensor, Config.image_angle[angle]))
                    print tensor.shape[0]
                    labels = np.ndarray(tensor.shape[0], dtype=int)
                    labels[0:tensor.shape[0]] = int(label)
                    labelList.append(labels)
            except Exception as e:
                print 'Unable to process data from {}, {}'.format(pickleFile, e)
        
    tensorFinal = np.concatenate(tensorList)
    labelFinal = np.concatenate(labelList)
    return tensorFinal, labelFinal
        

def tensor_to_matrix(tensor):
    '''
    '''
    return tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2])
    
    
    
    
    
    