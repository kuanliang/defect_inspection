from sklearn.externals import joblib
from dqlib.DataIO import load_queries
from dqlib.DataIO import extract_images_from_dir
from dqlib.Transform import extract_bottleneck_features


def load_and_predict(path_to_images, path_to_model):
    '''load model and predict images exist in specified directory
    
    Notes:
    
    Args:
        path_to_images (string): path to images ready for prediction       
        path_to_model (string): path to the model used for classification
    
    Return:
        predict_result (list): defect image prediction result
    
    '''
    # loading the model
    clf = joblib.load(path_to_model)
    paths_to_images_list = extract_images_from_dir(path_to_images)
    image_features_transformed = extract_bottleneck_features(paths_to_images_list)
    predict_result = clf.predict(image_features_transformed)
    
    print(predict_result)