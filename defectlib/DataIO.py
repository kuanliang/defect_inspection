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
                                
                                


def create_localTensors(path_to_local):
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
        print img_file
        if os.path.splitext(img_file)[1] == '.jp2':
            rgb_image = cv2.imread(os.path.join(path_to_local, img_file))
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            # print img_file
            # return rgb_image
            # print img_file
            # print gray_image.shape
            tensorList.append(gray_image)
            
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
    dataset = np.ndarray((len(tensorList), tensorList[0].shape[0], tensorList[0].shape[1]), dtype=np.float32)
    labels = np.ndarray((len(tensorList), 4), dtype=np.int32)
    for index, tensor in enumerate(tensorList):
        dataset[index,:,:] = tensor
        labels[index,:] = labelList[index]
        
        
    return dataset, labels
              