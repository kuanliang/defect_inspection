# Import the necessary packages
import numpy as np


	
def rgb2gray(rgb):
    '''rgb to gray image
    
    Notes:
    
    Args:
    
    Return:
    '''
    return np.dot(rgb, [0.299, 0.587, 0.114])