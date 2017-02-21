import matplotlib.pyplot as plt 
import cv2
import numpy as np

from defectlib.Transform import randomize


def display_tensor(tensor, label, sn):
    '''display images of all classes
    
    Notes:
    
    Argus:
        tensor: a tensor 
        label: related labels
    
    Return:
        None
    
    '''
    
    tensor, label, sn = randomize(tensor, label, sn)
    
    plt.figure(figsize=(10, 10))
    class_number = len(set(label))
    all_index = 1
    for class_index in set(label):
        if len(tensor[label == class_index]) < 4:
            col_num = len(tensor[label == class_index])
        else:
            col_num = 4
        for img_index in range(col_num):
            plt.subplot(class_number, 4, all_index)
            plt.axis('off')
            plt.imshow(tensor[label == class_index][img_index], cmap='gray')
            # plt.set_xticklabels([]*10)
            # plt.xlabel(sn[label == class_index][img_index])
            # plt.xlabel('test2')
            plt.title(sn[label == class_index][img_index])
            all_index += 1
    

def display_localTensor(tensor, label):
    
    
    plt.figure(figsize=(15, 7))
    
    tensor_num = tensor.shape[0]
    for index, tensor_index in enumerate(np.random.permutation(tensor_num)):
        if index < 9:
            img_rec = tensor[tensor_index].copy()
            cv2.rectangle(img_rec, tuple(label[tensor_index][0:2]), tuple(label[tensor_index][2:4]), (255, 0, 0), 10)
            plt.subplot(3, 3, index+1)
            plt.axis('off')
            plt.title('index {}'.format(tensor_index))
            plt.imshow(img_rec, cmap='gray')