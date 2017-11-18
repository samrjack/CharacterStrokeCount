'''
data_util.py
    This file contains many functions for loading data or parts of the data set
    as numpy arrays.
'''

import os
import numpy as np
from skimage.io import imread

'''
get_file_names
    Retrieves the names of all the files in the data set.

    list_of_stroke_counts - A list containing the stroke counts of
        the desired data. Files are filtered to contain only the specified
        stroke counts

    returns - A list of all the files in the data set
'''
def get_file_names(list_of_stroke_counts=False):
    file_names = os.listdir("../data/")
    if list_of_stroke_counts is False:
        return file_names
    desired_strokes = [str(c) for c in list_of_stroke_counts]
    return [file_name for file_name in file_names if get_file_stroke_count(file_name) in desired_strokes]

'''
get_pictures_with_stroke_count
    creates an numpy array containing the data of the specified
    stroke count character sets. 

    list_of_stroke_counts - A list of all the stroke count values to fetch.

    returns - A numpy array of flattened values for each picture.
'''
def get_pictures_with_stroke_count(list_of_stroke_counts=[]):
    files = get_file_names(list_of_stroke_counts)
    return get_pictures(files)

'''
get_pictures
    Fetches a picture's data from disk.

    file_list - list of files to import.

    returns - A numpy list containing flattened values for all
        the given pictures.
'''
def get_pictures(file_list):
    return np.array(list(map(import_photo, file_list)))

'''
import_photo
    gets a photo from memory, flattens it, and scales all the values to be
    between 0 and 1.

    file_name - the name of the picture to import.

    returns - a 1D numpy array containing the data for the given image.
'''
def import_photo(file_name):
    image = imread("../data/" + file_name)
    return image.flatten()/np.max(image)

'''
get_file_stroke_count
    Returns the stroke count of the character depicted in the file.
    Used so that if the name format chages, don't have to change every function.

    returns - The file's character's stroke count.
'''
def get_file_stroke_count(file_name):
    return file_name.split(".")[0]
