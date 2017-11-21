'''
data_util.py
    This file contains many functions for loading data or parts of the data set
    as numpy arrays.
'''

import os
import numpy as np
from skimage.io import imread

def get_file_names(list_of_stroke_counts=False):
    '''
    Retrieves the names of all the files in the data set.

    Parameters
    ----------
    list_of_stroke_counts : [int] 
        A list containing the stroke counts of
        the desired data. Files are filtered to contain only the specified
        stroke counts

    Returns
    -------
    [string]
        A list of all the files in the data set
    '''
    file_names = os.listdir("../data/")
    if list_of_stroke_counts is False:
        return file_names
    desired_strokes = [str(c) for c in list_of_stroke_counts]
    return [file_name for file_name in file_names if get_file_stroke_count(file_name) in desired_strokes]

def get_pictures_with_stroke_count(list_of_stroke_counts=[]):
    '''
    creates an numpy array containing the data of the specified
    stroke count character sets. 

    Parameters
    ----------
    list_of_stroke_counts : [int] 
        A list of all the stroke count values to fetch.

    Returns
    -------
    numpy.array(numpy.array(float))
    A numpy array of flattened values for each picture.
    '''
    files = get_file_names(list_of_stroke_counts)
    return get_pictures(files)

def get_pictures(file_list):
    '''
    Fetches a picture's data from disk.

    Parameters
    ----------
    file_list : [string]
        list of files to import.

    Returns
    -------
    numpy.array(numpy.array(float)) : len(file_list) x sizesize of file 
        all of the files representing rows in a large array.
    '''
    return np.array(list(map(import_photo, file_list)))

def import_photo(file_name):
    '''
    Gets a photo from memory, flattens it, and scales all the values to be
    between 0 and 1.

    Parameters
    ----------
    file_name : string
        The name of the picture to import.

    Returns
    -------
    1D numpy array
        Data for the given image.
    '''
    image = imread("../data/" + file_name)
    return image.flatten()/np.max(image)

def get_file_stroke_count(file_name):
    '''
    Returns the stroke count of the character depicted in the file.
    Used so that if the name format chages, don't have to change every function.

    Parameters
    ----------
    file_name : string
        A filename using standard file formating for this project.

    Returns
    -------
    int 
        The file's character's stroke count.
    '''
    return int(file_name.split(".")[0])
