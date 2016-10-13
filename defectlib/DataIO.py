from scipy import ndimage
import numpy as np
import pickle
import cv2
import os

pixel_depth = 255.0  # Number of levels per pixel.

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
    
    print image_files
    # from config import image shape information
    from Config import imageShapeDict
    # initialize numpy array for images
     
    # get image location from folder name
    print folder
    # test = folder.split('/')[-2]
    imageLoc = folder.split('/')[-1].split('_')[0]
    # print imageLoc
    # get angle of the camera from folder name
    imageAngle = folder.split('/')[-1].split('_')[1][1]
    # print imageAngle
    
    dataset = np.ndarray(shape=(len(image_files),
                         imageShapeDict[imageLoc][imageAngle]['height'], 
                         imageShapeDict[imageLoc][imageAngle]['width']), 
                         dtype=np.float32)
    
    num_images = 0
    
    for image in image_files:
        print image
        # print os.path.splitext(image)[1]
        if os.path.splitext(image)[1] == '.png':
            # print image
            try:
                # the image data
                image_file = os.path.join(folder, image)
                # read in the image to numpy array
                rgb_image = cv2.imread(image_file)
                # print image_data
                gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
                image_data = (gray_image.astype(float) - pixel_depth / 2) / pixel_depth
                
                # print image_data.shape
                # print dataset.shape
                
                dataset[num_images,:,:] = image_data
                num_images += 1
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
    dataset = dataset[0:num_images, :, :]
    
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    
    return dataset
    
def maybe_pickle(defect_folder, force=False):
    '''read in image files in tensor array and pickle it to specified directory
    
    Notes:
        the directory is like the following:
        defect_tensors/tp/9/tp_a9_c0/
        
    Args:
        defect_folder: the folder contain defect images
    
    Return:
        None
    
    '''
    # choose a angle and go to images 
    for defect_loc in os.listdir(defect_folder):
        if not defect_loc.startswith('.'):
            for defect_angle in os.listdir(os.path.join('.', defect_folder, defect_loc)):
                if not defect_angle.startswith('.'):
                    # initialize a list to include all tensor arrays
                    dataset_names = []
                    for defect_class in os.listdir(os.path.join('.', defect_folder, defect_loc, defect_angle)):
                        if not defect_class.startswith('.') and not defect_class.endswith('pickle'):
                            
                            pickleName = defect_class + '.pickle'
                            pathToPickle = os.path.join(defect_folder, defect_loc, defect_angle)
                            pathToImage =  os.path.join(defect_folder, defect_loc, defect_angle, defect_class)

                            # print pathToImage
                            # 
                            dataset = load_defects(pathToImage)
                            try:
                                with open(os.path.join(pathToPickle, pickleName), 'wb') as f:
                                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                            except Exception as e:
                                print('Unable to save data to', pickleName, ':', e)
                