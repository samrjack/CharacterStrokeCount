import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import sklearn as sk
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
    return image.flatten()/255

def show_weights_of_svd(stroke_counts, show=False):
    for stroke_count in stroke_counts:
        feature_array = get_pictures_with_stroke_count([stroke_count])
        if len(feature_array) < 1:
            continue
        U,s,V = np.linalg.svd(feature_array)
        title = "PCA components for stroke count of " + str(stroke_count)
        plt.figure(stroke_count,figsize=(13,10)).canvas.set_window_title(title)
        for i in range(1,10):
            if s[i] < 1e-10:
                break
            plt.subplot(3,3,i)
            plt.axis('off')
            plt.imshow(V[i-1,:].reshape(32,32))
            plt.gray()
            plt.title("weights " + str(i))
            plt.savefig("./figures/" + str(stroke_count) + ".pca.png")
    if(show):
        plt.show()
    plt.close('all')

def get_data_bar_graph(show=False):
    files = get_file_names()
    files = [c.split(".")[0] for c in files]
    files.sort()
    a = [(int(key),len(list(group))) for key, group in groupby(files)]
    a.sort()
    values = [num for num,_ in a]
    amount = [amt for _,amt in a]
    
    plt.figure(figsize=(13,10))
    plt.bar(values, amount, align='center', alpha=0.5)
    plt.xticks(values, values)
    plt.ylabel("Number of samples")
    plt.xlabel("Number of strokes")
    plt.title("Number of samples of each stroke count")

    plt.savefig("./figures/data_bar_graph.png")
    if(show):
        plt.show()
    plt.close('all')
    
#get_data_bar_graph()
