import os
import numpy as np
import pickle
import cv2
from imutils import rotate

import Config

from keras.utils import np_utils


def randomize(tensor, labels, sn):
    '''shuffle the tensor and its labels

    Notes:

    Args:
        tensor:
        labels:

    Return:
        shuffled_tensor:
        shuffled_labels:

    '''
    permutation = np.random.permutation(labels.shape[0])
    shuffled_tensor = tensor[permutation,:,:]
    shuffled_labels = labels[permutation]
    shuffled_sn = sn[permutation]

    return shuffled_tensor, shuffled_labels, shuffled_sn


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


def load_tensors(pickle_folder, width=256):
    '''load images as tensors from specified angle folder
    
    Note:
    
    Args:
    
    Return:

    '''
    tensorList = []
    labelList = []
    snList = []
    
    
    for pickleFile in os.listdir(pickle_folder):
        # print label
        # print pickleFile
        if pickleFile.endswith('pickle'):
            defect_loc = os.path.splitext(pickleFile)[0].split('_')[0]
            # print pickleFile
            label = int(os.path.splitext(pickleFile)[0].split('_')[-1].replace('c', ''))
            # print label
            camera_angle = os.path.splitext(pickleFile)[0].split('_')[1].replace('a', '')
            # print angle
            try:
                with open(os.path.join(pickle_folder,pickleFile), 'rb') as f:
                    # print 'path: {}'.format(os.path.join(pickle_folder, pickleFile))
                    tensor_sn = pickle.load(f)
                    tensor = tensor_sn[0]
                    sn_angle = [x + '/' + camera_angle for x in tensor_sn[1]]
                    # print tensor.shape
                    tensorList.append(align_tensor(tensor, Config.imageAngleDict[defect_loc.replace(' ', '_')][camera_angle], width=width))
                    # print tensor.shape[0]
                    labels = np.ndarray(tensor.shape[0], dtype=int)
                    labels[0:tensor.shape[0]] = int(label)
                    labelList.append(labels)
                    snList += sn_angle
            except Exception as e:
                print 'Unable to process data from {}, {}'.format(pickleFile, e, os.path.join(pickle_folder, pickleFile))

    tensorFinal = np.concatenate(tensorList)
    labelFinal = np.concatenate(labelList)
    snFinal = np.array(snList)
    return tensorFinal, labelFinal, snFinal


def load_tensors_all(angle_folder, width=256):
    '''load tensors from all angles
    
    Notes:

    Args:
        angle_folder (string): path to the directory file
        width (int): image size  
    Return:
        tensor_dict (dictionary): Python dictionary
        
    
    '''
    tensor_dict = {}
    directories_list = os.listdir(angle_folder)
    directories_list = [x for x in directories_list if not '.' in x]
    for angle_dir in directories_list:
        temp_dict = {}
        tensors, labels, sns = load_tensors(os.path.join(angle_folder, angle_dir))
        temp_dict['tensors'] = tensors
        temp_dict['labels'] = labels
        temp_dict['sn'] = sns
        tensor_dict[angle_dir] = temp_dict
        
        
        
    return tensor_dict
    


def tensor_to_matrix(tensor):
    '''
    '''
    return tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2])


def combine_shuffle_tensors(*tensorLabels):
    '''combine different tensors and shuffle them

    Notes:

    Args:

    Return:

    '''
    # print type(tensorLabels[0])
    if type(tensorLabels[0]) is tuple:
        
        # print 'get'
        tensorList = []
        labelList = []
    
        tensor_length = 0
    
        for tensor, label in tensorLabels:
            tensor_length += tensor.shape[0]
    
        print 'the final tensor should be {}'.format(tensor_length)
    
        height = tensor.shape[1]
        width = tensor.shape[2]
        # initialize empth tensor and label
        # combined_tensor = np.ndarray(shape=(tensor_length, height, width), dtype=np.float32)
        # combined_label = np.ndarray(tensor_length, dtype=int)
    
        for tensor, label in tensorLabels:
            tensorList.append(tensor)
            labelList.append(label)
    
        final_tensor = np.concatenate(tensorList)
        final_label = np.concatenate(labelList)
    
        shuffled_tensor, shuffled_label = randomize(final_tensor, final_label)
    
    elif type(tensorLabels[0]) is dict:
        
        tensor_dict = tensorLabels[0]
        
        tensorList = []
        labelList = []
        snList = []
        
        tensor_length = 0
        
        for angle in tensorLabels[0]:
            tensor_length += len(tensor_dict[angle]['labels'])
            tensorList.append(tensor_dict[angle]['tensors'])
            labelList.append(tensor_dict[angle]['labels'])
            snList.append(tensor_dict[angle]['sn'])
            
        final_tensor = np.concatenate(tensorList)
        final_label = np.concatenate(labelList)
        final_sn = np.concatenate(snList)
        
        print 'the final tensor should be {}'.format(tensor_length)
        
        shuffled_tensor, shuffled_label, shuffled_sn = randomize(final_tensor, final_label, final_sn)
            
    
    return shuffled_tensor, shuffled_label, shuffled_sn
    
def keras_transform(original_tensors, original_labels):
    '''transform tensors to keras format and OHE labels
    
    Notes:
        the following Keras setting for tensor format is (index, height, width, 1)
        but it may changed, check latest Keras document if something wrong
    
    Args:
        original_tensors (3 dim numpy array):
        original_labels (1 dim numpy array):
    
    Return:
        keras_tensors (4 dim numpy array)
        keras_labels (n dim numpy array)
    '''
    keras_tensors = original_tensors.reshape(original_tensors.shape[0],
                                             original_tensors.shape[1],
                                             original_tensors.shape[2],
                                             1)
    keras_label = np_utils.to_categorical(original_labels)
    
    return keras_tensors, keras_label