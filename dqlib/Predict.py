from sklearn.externals import joblib
from DataIO import load_queries

def load_predict(path_to_images, path_to_model):
    '''load model and predict images exist in specified directory
    
    Notes:
    
    Args:
    
    Return:
    
    '''
    
    # loading the model
    
    clf = joblib.load(path_to_model)
    images_data = load_queries(path_to_images)
    
    predict_result = clf.predict(images_data)
    
    print predict_result
    
    return predict_result