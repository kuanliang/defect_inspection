import os
import numpy as np
import pickle
import cv2
from defectlib.imutils import rotate

import defectlib.Config

from keras.utils import np_utils
import pandas as pd


# import for transfer leraning
import os
import re
import glob

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import sklearn
import pickle

from defectlib.DataIO import extract_images_from_dir


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


def align_tensor(tensor, angle, width=256):
    '''align and resize the tensor to univeral (256*256) scale

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

        empty_tensor[index] = rotated_image

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
            except Exception as e:
                print('Unable to process data from {}, {}'.format(pickleFile, e, os.path.join(pickle_folder, pickleFile)))
                
            tensor = tensor_sn[0]
            sn_angle = [x + '/' + camera_angle for x in tensor_sn[1]]
            print(tensor.shape)
            tensorList.append(align_tensor(tensor, Config.imageAngleDict[defect_loc.replace(' ', '_')][camera_angle], width=width))
            # print tensor.shape[0]
            labels = np.ndarray(tensor.shape[0], dtype=int)
            labels[0:tensor.shape[0]] = int(label)
            labelList.append(labels)
            snList += sn_angle


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
        tensors, labels, sns = load_tensors(os.path.join(angle_folder, angle_dir), width=width)
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
        snList = []
    
        tensor_length = 0
    
        for tensor, label, sn in tensorLabels:
            tensor_length += tensor.shape[0]
    
        print('the final tensor should be {}'.format(tensor_length))
    
        height = tensor.shape[1]
        width = tensor.shape[2]
        # initialize empth tensor and label
        # combined_tensor = np.ndarray(shape=(tensor_length, height, width), dtype=np.float32)
        # combined_label = np.ndarray(tensor_length, dtype=int)
    
        for tensor, label, sn in tensorLabels:
            tensorList.append(tensor)
            labelList.append(label)
            snList.append(sn)
    
        final_tensor = np.concatenate(tensorList)
        final_label = np.concatenate(labelList)
        final_sn = np.concatenate(snList)
    
        shuffled_tensor, shuffled_label, shuffled_sn = randomize(final_tensor, final_label, final_sn)
    
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
        
        
        print('the final tensor should be {}'.format(tensor_length))
        
        
        shuffled_tensor, shuffled_label, shuffled_sn = randomize(final_tensor, final_label, final_sn)
        
    
    label_summary = list(set(shuffled_label))
    label_summary.sort()
    
    for index, item in enumerate(pd.Series(shuffled_label).value_counts()):
        print('number of class {}: {}'.format(label_summary[index], item))
        print('\tnumber of SN: {}'.format(len(set([sn.split('/')[0] 
                        for sn in shuffled_sn[shuffled_label == label_summary[index]]]))))
        
    return shuffled_tensor, shuffled_label, shuffled_sn
    
def keras_transform(original_tensors, original_labels, image_dim_ordering='tf'):
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
    if image_dim_ordering == 'tf':
        keras_tensors = original_tensors.reshape(original_tensors.shape[0],
                                                 original_tensors.shape[1],
                                                 original_tensors.shape[2],
                                                 1)
    elif image_dim_ordering == 'th':
                keras_tensors = original_tensors.reshape(
                                                 original_tensors.shape[0],
                                                 1,
                                                 original_tensors.shape[1],
                                                 original_tensors.shape[2])
                                                 
    keras_label = np_utils.to_categorical(original_labels)
    
    return keras_tensors, keras_label
    
def remove_sn(tensors, labels, sns, remove_sn):
    '''
    '''
    sns_only = np.array([x.split('/')[0] for x in sns])
    mask = sns_only != remove_sn
    tensors_removed = tensors[mask]
    labels_removed = labels[mask]
    sns_removed = sns[mask]
    
    return tensors_removed, labels_removed, sns_removed

def remain_sn(tensors, labels, sns, remain_sn):
    '''
    '''
    sns_remained = np.array([x.split('/')[0] for x in sns])
    masks = sns_remained == remain_sn
    tensors_remained = tensors[masks]
    labels_remained = labels[masks]
    sns_remained = sns[masks]
    
    return tensors_remained, labels_remained, sns_remained
    
# transfer leraning: create Graph
def create_graph(model_path):
    '''create graph from the model specified directory path
    
    Notes: 
    
    Args:
        model_path (string): directory path to the downloaded model
    
    Return:
        None
    '''
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        
        
def extract_bottleneck_features(list_images):
    '''extract buttleneck features from a list of images
    
    Notes:
        
    
    Args:
        list_images (list): a list of path_to_images
    
    Return:
        features (numpy array): an 2 dimensional numpy array,
                                each row represents a transformed feature of an image
        
    '''
    # transformed feature
    nb_features = 2048
    
    # initial feature numpy array
    features = np.empty((len(list_images),nb_features))
    
    labels = []
    
    # specified the inception model
    create_graph('./inception_dec_2015/tensorflow_inception_graph.pb')
    
    
    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        #return next_to_last_tensor
        for ind, image in enumerate(list_images):
            if (ind%100 == 0):
                print('Processing %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()
            # print type(image_data)
            # print image_data.shape
            predictions = sess.run(next_to_last_tensor,
                {'DecodeJpeg/contents:0': image_data}
            )
            features[ind,:] = np.squeeze(predictions)
            # labels.append(re.split('_\d+', image.split('/')[1])[0])
            # print labels

    return features


def extract_normal_features():
    '''extract normal features from images
    
    Notes:
    
    Args:
    
    Return:
    
    '''

    
    return features

def extract_bnfeatures_from_angle(path, comb):
    '''extract features from specified angle path
    
    Notes:
    
    Args:
        path (string): path to the specified directory
        comb (boolean): whether use combination image
        
    Return:
        features_final (numpy array): array containing features 
        labelss_final (numpy array): array containing associated labels
        sns_final (numpy array): array containing associated S/N
        
    '''
    
    angle_dir_names = [dir_names for dir_names in os.listdir(path) if os.path.isdir(os.path.join(path, dir_names))]
    
    # initialize empty list for storing objects
    features_in_angle = []
    labels_in_angle = []
    sns_in_angle = []
    images_in_angle = []
    
    
    for class_dir in angle_dir_names:
        
        labels_empty = []
        # use extract_images_from_dir to extract image list
        images_list = extract_images_from_dir(os.path.join(path, class_dir), comb=comb)
        print('there are {} images inside {}'.format(len(images_list), class_dir))
        # transform feature from exist model
        features = extract_bottleneck_features(images_list)
        labels_empty.append(class_dir.split('_')[-1][1:])
        labels = labels_empty * len(images_list)
        if comb:
            sns = [os.path.basename(x).split('_')[2] for x in images_list]
        else:
            sns = [os.path.basename(x).split('_')[0].split(' ')[0] for x in images_list]
        # 
        features_in_angle.append(features)
        labels_in_angle.append(labels)
        sns_in_angle.append(sns)
        images_in_angle.append(images_list)
        
    
    features_final = np.concatenate(features_in_angle)
    labels_final = np.concatenate(labels_in_angle)
    sns_final = np.concatenate(sns_in_angle)
    images_final = np.concatenate(images_in_angle)
    
    return features_final, labels_final, sns_final, images_final
    
def extract_bnfeatures_from_defect(path, comb=False):
    '''extract features from images within specified defect directory
    
    Notes:
        
        
    Args:
        path (string): path to the specified defect directory
    
    Return:
        features_all (numpy array): 
        labels_all (numpy array):
        sns_all (numpy array):
    
    '''
    
    # initialize the empty list for future object storage
    features_list = []
    labels_list = []
    sns_list = []
    images_list = []
    
    # iterate through each angle directory
    for angle_dir in [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]:
        
        angle_path = os.path.join(path, angle_dir)
        
        # extract features from specified angle directory
        features, labels, sns, images = extract_bnfeatures_from_angle(angle_path, comb=comb)
        
        # append extracted numpy array object (features, labels, sns, images to the list)
        features_list.append(features)
        labels_list.append(labels)
        sns_list.append(sns)
        images_list.append(images)
    # np.concatenate the list of numpy arrays
    features_all = np.concatenate(features_list)
    labels_all = np.concatenate(labels_list)
    sns_all = np.concatenate(sns_list)
    images_all = np.concatenate(images_list)
    
    return features_all, labels_all, sns_all, images_all
    
    

def extract_query_bnfeatures_from_defect(path, comb=False):
    '''extract features from images within specified defect directory
    
    Notes:
        
        
    Args:
        path (string): path to the specified defect directory
    
    Return:
        features_all (numpy array): 
        labels_all (numpy array):
        sns_all (numpy array):
    
    '''
    
    # initialize the empty list for future object storage
    features_list = []
    sns_list = []
    images_list = []
    
    # iterate through each angle directory
    for angle_dir in [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]:
        
        images = extract_images_from_dir(os.path.join(path, angle_dir), comb=comb)
        
        # extract features from specified angle directory
        features = extract_bottleneck_features(images)
        
        # append extracted numpy array object (features, labels, sns, images to the list)
        features_list.append(features)
        # sns_list.append(sns)
        images_list.append(images_list)
    # np.concatenate the list of numpy arrays
    features_all = np.concatenate(features_list)
    # sns_all = np.concatenate(sns_list)
    images_all = np.concatenate(images_list)
    
    return features_all, images_all