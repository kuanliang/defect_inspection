from scipy import ndimage
import numpy as np
import pandas as pd
import pickle
import cv2
import os
import random
from tqdm import tqdm, trange
import matplotlib.image as mpimg

pixel_depth = 256.0  # Number of levels per pixel.

def load_defects(folder):
    '''Load defect images as a tensor dataset of specific angle (vision), specific defect
    
    Notes:
    
    Args:
        folder: 
        angle: 
    
    Return:
        dataset: a dataset contain image arrays where first index is the number of images
    
    '''
    # images file names in a list
    image_files = os.listdir(folder)
    image_files = [x for x in image_files if not 'DS_' in x]
    image_files = [x for x in image_files if not 'ipy' in x]
    
    # print 'This is {}'.format(image_files)
    # from config import image shape information
    # from Config import imageShapeDict
    # initialize numpy array for images
     
    # get image location from folder name
    # print folder
    # test = folder.split('/')[-2]
    imageLoc = folder.split('/')[-1].split('_')[0]
    # print imageLoc
    # get angle of the camera from folder name
    imageAngle = folder.split('/')[-1].split('_')[1][1]
    # print imageAngle
    if len(image_files) > 0:
        # print image_files
        # get image shape
        random_image = random.choice(image_files)
        image_loaded = mpimg.imread(os.path.join(folder, random_image))
        # initialize numpy array
        dataset = np.ndarray(shape=(len(image_files),
                                    #imageShapeDict[imageLoc][imageAngle]['height'], 
                                    #imageShapeDict[imageLoc][imageAngle]['width']), 
                                    image_loaded.shape[0],
                                    image_loaded.shape[1]),
                                    dtype=np.float32)
        # sn_array = np.ndarray(shape=(len(image_files)))
        sn_list = []
    else:
        dataset = np.ndarray(shape=(len(image_files),
                                    #imageShapeDict[imageLoc][imageAngle]['height'], 
                                    #imageShapeDict[imageLoc][imageAngle]['width']), 
                                    256,
                                    256),
                                    dtype=np.float32)
        sn_list = []
    
    
    # initialize the dictionary
    num_images = 0
    
    for image in image_files:
        # print os.path.splitext(image)[1]
        # ignore Recommbination images
        if os.path.splitext(image)[1] == '.png' or os.path.splitext(image)[1] == '.jpg' and 'Recom' in image:
            # print image
            # without Recombination 
            # sn_nb = os.path.splitext(image)[0].split()[0]
            # with Recombination
            sn_nb = os.path.splitext(image)[0].split('_')[2]
            try:
                # the image data
                image_file = os.path.join(folder, image)
                # read in the image to numpy array
                rgb_image = mpimg.imread(image_file)
                # print image_data
                gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
                image_data = (gray_image.astype(float) - pixel_depth / 2) / pixel_depth
                
                # print image_data.shape
                # print dataset.shape
                
                dataset[num_images,:,:] = image_data
                # sn_raray[num_images] = sn_nb
                sn_list.append(sn_nb)
                num_images += 1
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
    dataset = dataset[0:num_images, :, :]
    
    # print('Full dataset tensor:', dataset.shape)
    # print('Mean:', np.mean(dataset))
    # print('Standard deviation:', np.std(dataset))
    
    return dataset, sn_list
    
def maybe_pickle(defect_folder, force=False):
    '''read in image files in tensor array and pickle it to specified directory
    
    Notes:
        the directory is like the following:
        defect_tensors/tp/9/tp_a9_c0/
        
    Args:
        defect_folder: the folder contain defect images, e.g., './defect_tensors/'
    
    Return:
        None
    
    '''
    # choose a angle and go to images 
    for defect_loc in tqdm(os.listdir(defect_folder)):
        if not defect_loc.startswith('.'):
            # print 'processing {}'.format(defect_loc)
            for defect_angle in os.listdir(os.path.join('.', defect_folder, defect_loc)):
                if not defect_angle.startswith('.') and not defect_angle.endswith('csv'):
                    # print 'processing {}'.format(defect_angle)
                    # initialize a list to include all tensor arrays
                    dataset_names = []
                    for defect_class in os.listdir(os.path.join('.', defect_folder, defect_loc, defect_angle)):
                        if not defect_class.startswith('.') and not defect_class.endswith('pickle'):
                            # print 'process {}'.format(defect_class)    
                            pickleName = defect_class + '.pickle'
                            pathToPickle = os.path.join(defect_folder, defect_loc, defect_angle)
                            pathToImage =  os.path.join(defect_folder, defect_loc, defect_angle, defect_class)

                            # print pathToImage
                            # 
                            dataset, sn_list = load_defects(pathToImage)
                            try:
                                with open(os.path.join(pathToPickle, pickleName), 'wb') as f:
                                    pickle.dump((dataset, sn_list), f, pickle.HIGHEST_PROTOCOL)
                            except Exception as e:
                                print('Unable to save data to', pickleName, ':', e)
                                
                                


def create_localTensors(path_to_local, gray=True):
    '''create tensors for localizers
    
    Notes:
    
    Args:
    
    Return:
    
    '''
    
    # read in the coordinate information with Pandas
    pathToCoor = os.path.join(path_to_local, 'For_Andy', 'Data.txt')
    coorDf = pd.read_csv(pathToCoor, names=['file_name', 'coordinate'], sep='\t')
    coorDf['file_name'] = [os.path.splitext(x)[0] for x in coorDf['file_name']]    
    
    # return coorDf
    # return coorDf
    # initialize a tensorList and labelist
    tensorList = []
    labelList = []
    
    
    # return coorDf
    # loop through images and store them into numpy array
    for img_file in os.listdir(path_to_local):
        # print img_file
        if os.path.splitext(img_file)[1] == '.jp2':
            rgb_image = cv2.imread(os.path.join(path_to_local, img_file))
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            # print img_file
            # return rgb_image
            # print img_file
            # print gray_image.shape
            if gray == True:
                tensorList.append(gray_image)
            else:
                tensorList.append(rgb_image)
            # extract coordinate information (top_left_x, top_left_y, right_bottom_x, right_bottom_y)
            # try:
            
            image_main = os.path.splitext(os.path.splitext(img_file)[0])[0]
            # print image_main
            coordinate_tuple = coorDf[coorDf['file_name'] == image_main]['coordinate'].values[0].split()
            
            # print coordinate_tuple
            # except:
            # print "can't access coordinate infor from coordinate dataframe"
                
            labelList.append(coordinate_tuple)
    
    # return tensorList
    
    # tensorFinal = np.concatenate(tensorList)
    # labelFinal = np.concatenate(labelList)
    if gray == True:
        dataset = np.ndarray((len(tensorList), tensorList[0].shape[0], tensorList[0].shape[1]), dtype=np.float32)    
    else:
        dataset = np.ndarray((len(tensorList), tensorList[0].shape[0], tensorList[0].shape[1], 3), dtype=np.uint8) 
    labels = np.ndarray((len(tensorList), 4), dtype=np.float32)
    
    # get width and height of the image
    width = tensorList[0].shape[1]
    height = tensorList[0].shape[0]
    print width
    print height
    
    for index, tensor in enumerate(tensorList):
        if gray == True:
            dataset[index,:,:] = tensor
        else:
            dataset[index,:,:,:] = tensor
        # print labelList[index]
        # print np.array([width, height, width, height], dtype=np.float32)
        # labels[index,:] = np.array([float(i) for i in labelList[index]], dtype=np.float32) / np.array([width, height, width, height], dtype=np.float32) 
        labels[index,:] = np.array(labelList[index])
        
    return dataset, labels