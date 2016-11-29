from keras.models import Sequential
from keras.layers.core import Flatten, Dropout, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from Transform import keras_transform

from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

def escape_rate(true, predict):
    '''calculate escape rate 
    
    '''
    return len(np.where(predict == 0)) / len(predict)
    

def make_model(nb_classes, image_dim_ordering='tf', input_shape = (128, 128)):
    '''make the model via Keras
    
    
    '''
    if image_dim_ordering == 'tf':
        input_shape = (input_shape[0], input_shape[1], 1)
    elif image_dim_ordering == 'th':
        input_shape = (1, input_shape[0], input_shape[1])
        
    model = Sequential()

    # Convolution2D(number_filters, row_size, column_size, input_shape=(number_channels, img_row, img_col))
    
    model.add(Convolution2D(6, 5, 5, input_shape=input_shape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(120, 5, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
    
    return model
    
def train_model(model, (train_data, train_labels), (val_data, val_labels), nb_epoch=10, batch_size=30):
    '''
    '''
    model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=np_epoch,
              verbose=1, validation_data=(val_data, val_label))
    score = model.evaluate(val_data, val_labels, verbose=1)
    
    
def GroupKFold_modeling(tensors, labels, sn, nb_classes, batch_size=30, nb_epoch=10, input_shape=(128, 128)):
    '''use K-fold iterator with non-overlapping groups to do modeling
    
    Notes: The same group (S/N) will not appear in two different folds (the number of distinct groups has to be at least )
    
    Args:
        tensos (3-d numpy array): images
        labels (1-d numpy array): labels
        sn (1-d numpy array): serial numbers 
        nb_classes: the class of number being predicted
        batch_size: batch size
        nb_epoch: number of epochs
    
    Return:
        
    
    '''
    # generate non-overlapping K-fold iterator 
    sn_only = np.array([x.split('/')[0] for x in sn])
    nb_group = len(set(sn_only))
    group_kfold = GroupKFold(n_splits=nb_group)
    
    models = []
    
    accuracy_escape_list = []
    
    
    
    # sum_score = 0
    accuracy_dict = {}
    logloss_dict = {}
    escape_dict = {}
    nb_label = {}
    for label in set(labels):
        accuracy_dict[label] = 0
        logloss_dict[label] = 0
        escape_dict[label] = 0
        nb_label[label] = 0
    
    
    
    
        
    nb_model = 0
    for train_index, test_index in group_kfold.split(tensors, labels, sn_only):
        
        nb_model += 1
        
        label_val = set(labels[test_index])
        sn_val = set(sn_only[test_index])
        print 'Model {}'.format(nb_model)
        print 'the label of validation image: {}'.format(list(label_val)[0])
        print 'the s/n of validation image: {}'.format(list(sn_val)[0])
        # plt.imshow(tensors[test_index][0])
        tensors_k, labels_k = keras_transform(tensors, labels)
        tensors_train, labels_train = tensors_k[train_index], labels_k[train_index]
        tensors_val, labels_val = tensors_k[test_index], labels_k[test_index]
    
        model = make_model(nb_classes=nb_classes, input_shape=input_shape)
        model.fit(tensors_train, labels_train, batch_size=30, nb_epoch=nb_epoch, verbose=1,
                  validation_data=(tensors_val, labels_val))
                  
        predictions_val = model.predict(tensors_val)
        predictions_val_class = model.predict_classes(tensors_val)
        
        # return labels_val, predictions_val_class
        
        score = log_loss(labels_val, predictions_val)
        # print 
        # return np.array([np.argmax(x) for x in labels_val])
        accuracy = accuracy_score(np.array([np.argmax(x) for x in labels_val]), predictions_val_class)
        print('Sore log_loss: ', score)
        print('Accuracy:', accuracy)
        
        # if the class is not normal
        escape_rate = len(np.where(predictions_val_class == 0)[0]) / float(len(predictions_val_class))
        if list(label_val)[0] != 0:
            print 'Escape rate: {}'.format(escape_rate)
        
        accuracy_dict[list(label_val)[0]] += accuracy
        logloss_dict[list(label_val)[0]] += score
        escape_dict[list(label_val)[0]] += escape_rate
        
        # accumulate number of labels count
        nb_label[list(label_val)[0]] += 1
        
        # append accuracy_all_list and escape_all_list
        # [label, sn, accuracy, escape]
        accuracy_escape_list.append((list(label_val)[0], list(sn_val)[0], accuracy, escape_rate))
        
        models.append(model)
        
    accuracy_avg = {}
    logloss_avg = {}
    escape_avg = {}
    # take average of accuracy and logloss 
    for accuracy, logloss, escape in zip(accuracy_dict.items(), logloss_dict.items(), escape_dict.items()):
        accuracy_avg[accuracy[0]] = accuracy[1] / nb_label[accuracy[0]]
        logloss_avg[logloss[0]] = logloss[1] / nb_label[logloss[0]]
        escape_avg[escape[0]] = escape[1] / nb_label[escape[0]]
    
    # info_string = 'loss_' + str(score) + '_folds_' + str(n_splits) + '_ep_' + str(nb_epoch)
        
    # score = sum_score / len(labels_val)
    
    return accuracy_avg, logloss_avg, escape_avg, accuracy_escape_list, models

def KFold_modeling(tensors, labels, nb_classes=3, n_splits=3, batch_size=30, nb_epoch=10):
    '''use k-fold iterator to do modeling
    
    Notes:
    
    Args:
        tensors (numpy array): image array
        labels (numpy array): label array
        n_splits (int): number of fold for cross validation
        batch_size (int): number of records trainined per batch
        nb_epoch (int): number of epochs
    
    Return:
        info_string (string): cross validation result information
        models (list): models stored in list
        
    
    '''
    models = []
    sum_score = 0
    
    
    # initialize an kfold iterator
    kf = KFold(n_splits=n_splits, shuffle=True)
    
    
    for train_index, test_index in tqdm(kf.split(tensors)):
    
        train_tensors, train_labels = keras_transform(tensors[train_index], labels[train_index])
        val_tensors, val_labels = keras_transform(tensors[test_index], labels[test_index])
    
        model = make_model(nb_classes=nb_classes)
        model.fit(train_tensors, train_labels, batch_size=30, nb_epoch=nb_epoch, verbose=1,
                  validation_data=(val_tensors, val_labels))
                  
        predictions_valid = model.predict(val_tensors)
        
        score = log_loss(val_labels, predictions_valid)
        print('Sore log_loss: ', score)
        sum_score += score
        models.append(model)
    
    info_string = 'loss_' + str(score) + '_folds_' + str(n_splits) + '_ep_' + str(nb_epoch)
        
    score = sum_score / len(val_labels)
        
    return info_string, models
    
def random_modeling(tensors, labels, nb_classes, batch_siz=30, nb_epoch=10, random_state=50):
    '''split dataset to 80% training 20% testing for modeling
    
    Notes:
    
    Args:
        tensors (numpy array): image array
        labels (numpy array): label array
        n_splits (int): number of fold for cross validation
        batch_size (int): number of records trainined per batch
        nb_epoch (int): number of epochs
        random_state (int): random state number
    
    Return:
        model
    
    '''
    # split the input to train test sets
    tensors_train, tensors_val, labels_train, labels_val = train_test_split(tensors, labels, test_size=0.3,
                                                                              random_state=random_state)
    # transform to keras input
    tensors_train_k, labels_train_k = keras_transform(tensors_train, labels_train)
    tensors_val_k, labels_val_k = keras_transform(tensors_val, labels_val)
    model = make_model(nb_classes=3)
    model.fit(tensors_train_k, labels_train_k, batch_size=30, nb_epoch=nb_epoch, verbose=1, 
              validation_data=(tensors_val_k, labels_val_k))
              
    return model
    
    