import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
from skimage.io import imread
from itertools import groupby

def get_file_names(list_of_stroke_counts=False):
    file_names = os.listdir("../data/")
    if list_of_stroke_counts is False:
        return file_names
    desired_strokes = [str(c) for c in list_of_stroke_counts]
    return [file_name for file_name in file_names if file_name.split(".")[0] in desired_strokes]

def get_pictures_with_stroke_count(list_of_stroke_counts=[]):
    files = get_file_names(list_of_stroke_counts)
    return np.array(list(map(import_photo, files)))

def import_photo(file_name):
    image = imread("../data/" + file_name)
    return image.flatten()

def get_data_bar_graph():
    files = get_file_names()
    files = [c.split(".")[0] for c in files]
    files.sort()
    a = [(int(key),len(list(group))) for key, group in groupby(files)]
    a.sort()
    values = [num for num,_ in a]
    amount = [amt for _,amt in a]
    
    plt.bar(values, amount, align='center', alpha=0.5)
    plt.xticks(values, values)
    plt.ylabel("Number of samples")
    plt.xlabel("Number of strokes")
    plt.title("Number of samples of each stroke count")

    plt.show()
    
#get_data_bar_graph()
