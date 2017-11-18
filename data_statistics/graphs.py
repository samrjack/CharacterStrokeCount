'''
graphs.py
    This file contains many functions that create graphs and other visuals
    used to view the character set and visualize its distribution or 
    characteristics
'''

import matplotlib.pyplot as plt
import numpy as np
import data_util as du
from itertools import groupby

'''
show_weights_of_svd
    Creats pictures (stored in the figures folder) of the first 9 (or fewer)
    principal components of the given stroke counts. These pictures show the
    strength of each pixel in the 32x32 images.

    stroke_count - A list of all the stroke counts to display. Each stroke count
        is put into its own file and shown in its own figure window.
        ex: [4,6,8] shows pca of characters with 4, 6, and 8 stokes.

    show - Shows the picture in the display. Default action is to only save
        picture to a file.

    return - This function has no return value.
'''
def show_weights_of_svd(stroke_counts, show=False):
    for stroke_count in stroke_counts:
        feature_array = du.get_pictures_with_stroke_count([stroke_count])
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

'''
show_data_bar_graph
    Creates a bar graph (stored in a file in the figures folder) showing
    the data distribution of the test character set.

    show - Shows the graphy in the display. Default action is to only save
        picture to a file.

    return - This function has no return value.
'''
def show_data_bar_graph(show=False):
    files = du.get_file_names()
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
