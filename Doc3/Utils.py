from PIL import Image
import os


def png_to_jpg(path):
    '''transform png files to jpeg files

    Notes:

    Args:

    Return:

    '''
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.jpg'):
                im = Image.open(os.path.join(dirpath, filename))
                im.save(os.path.join(dirpath, filename), "JPEG")
