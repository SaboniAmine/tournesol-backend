from os import makedirs
from .plots import plot_metrics 

PATH_PLOTS = "ml/plots/"
makedirs(PATH_PLOTS, exist_ok=True)

def licch_stats(licch):
    ''' gives some statistics about licchavi object '''
    licch.check() # some tests
    h = licch.history
    print("nb_nodes", licch.nb_nodes)
    licch.stat_s()  # print stats on s parameters
    plot_metrics([h], path=PATH_PLOTS)

