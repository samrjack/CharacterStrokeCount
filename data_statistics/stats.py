import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
from itertools import groupby

def get_data_bar_graph():
    files = os.listdir("../data/")
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
    
get_data_bar_graph()
