import matplotlib.pyplot as plt 


def display_tensor(tensor, label):
    '''display images of all classes
    
    Notes:
    
    Argus:
        tensor: a tensor 
        label: related labels
    
    Return:
        None
    
    '''
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
            plt.imshow(tensor[label == class_index][img_index])
            all_index += 1
    
    
    
    
    