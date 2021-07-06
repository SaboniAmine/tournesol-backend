import matplotlib.pyplot as plt
import numpy as np 
from copy import deepcopy

INTENS = 0.4
METRICS = ({"lab":"fit", "ord": "Training Loss", "f_name": "loss"}, 
           {"lab":"gen", "ord": "Training Loss", "f_name": "loss"}, 
           {"lab":"reg", "ord": "Training Loss", "f_name": "loss"}, 
           {"lab":"acc", "ord": "Accuracy", "f_name": "acc"}, 
           {"lab":"l2_dist", "ord": "l2 norm", "f_name": "l2dist"}, 
           {"lab":"l2_norm", "ord": "l2 norm", "f_name": "l2dist"}, 
           {"lab":"grad_sp", "ord": "Scalar Product", "f_name": "grad"}, 
           {"lab":"grad_norm", "ord": "Scalar Product", "f_name": "grad"}
           )

def get_style():
    '''gives different line styles for plots'''
    l = ["-","-.",":","--"]
    for i in range(10000):
        yield l[i % 4]

def get_color():
    '''gives different line styles for plots'''
    l = ["red","green","blue","grey"]
    for i in range(10000):
        yield l[i % 4]

STYLES = get_style() # generator for looping styles
COLORS = get_color()

def title_save(title=None, path=None, suff=".png"):
    ''' add title and save plot '''
    if title is not None:   
        plt.title(title)
    if path is not None:
        plt.savefig(path + suff)

def legendize(y):
    ''' label axis of plt plot '''
    plt.xlabel("Epochs")
    plt.ylabel(y)
    plt.legend()

# def clean_dic(dic):
#     ''' replace some values by more readable ones '''
#     if "opt" in dic.keys():
#         dic = deepcopy(dic)
#         op = dic["opt"]
#         dic["opt"] = "Adam" if op == optim.Adam else "SGD" if op == optim.SGD 
#                                                 else None
#     return dic

# def get_title(conf, ppl=4):
#     ''' converts a dictionnary in str of approriate shape 
#         ppl : parameters per line
#     '''
#     title = ""
#     c = 0 # enumerate ?
#     for key, val in clean_dic(conf).items(): 
#         c += 1
#         title += "{}: {}".format(key,val)
#         title += " \n" if (c % ppl) == 0 else ', '
#     return title[:-2]

def means_bounds(arr):
    '''  
    arr: 2D array of values (one line is one run)

    Returns:
    - array of means
    - array of (mean - var)
    - array of (mean + var)
    '''
    means = np.mean(arr, axis=0)
    var = np.var(arr, axis = 0) 
    low, up = means - var, means + var
    return means, low, up


# ----------- to display multiple accuracy curves on same plot -----------
def add_acc_var(arr, label):
    ''' from array add curve of accuracy '''
    acc = arr[:,3,:]
    means, low, up = means_bounds(acc)
    epochs = range(1, len(means) + 1)
    plt.plot(epochs, means, label=label, linestyle=next(STYLES))
    plt.fill_between(epochs, up, low, alpha=0.4)

def plot_runs_acc(l_runs, title=None, path=None, **kwargs):
    ''' plot several acc_var on one graph '''
    arr = np.asarray(l_runs)
    l_param = get_possibilities(**kwargs) # for legend
    # adding one curve for each parameter combination (each run)
    for run, param in zip(arr, l_param): 
        add_acc_var(run, param)
    plt.ylim([0,1])
    plt.grid(True, which='major', linewidth=1, axis='y', alpha=1)
    plt.minorticks_on()
    plt.grid(True, which='minor', linewidth=0.8, axis='y', alpha=0.8)
    legendize("Test Accuracy")
    title_save(title, path, suff=".png")
    plt.show()

# ------------- utility for what follows -------------------------
def plot_var(l_hist, l_idx):
    ''' add curve of asked indexes of history to the plot '''
    arr_hist = np.asarray(l_hist)
    epochs = range(1, arr_hist.shape[2] + 1)
    for idx in l_idx:
        vals = arr_hist[:,idx,:]
        vals_m, vals_l, vals_u = means_bounds(vals)
        style, color = next(STYLES), next(COLORS)
        plt.plot(epochs, vals_m,    label=METRICS[idx]["lab"], 
                                    linestyle=style, 
                                    color=color)
        plt.fill_between(epochs, vals_u, vals_l, alpha=INTENS, color=color)

def plotfull_var(l_hist, l_idx, title=None, path=None, show=True):
    ''' plot metrics asked in -l_idx and save if -path provided '''
    plot_var(l_hist, l_idx)
    idx = l_idx[0]
    legendize(METRICS[idx]["ord"])
    title_save(title, path, suff=" {}.png".format(METRICS[idx]["f_name"]))
    if show: 
        plt.show()

# ------- groups of metrics on a same plot -----------
def loss_var(l_hist, title=None, path=None):
    ''' plot losses with variance from a list of historys '''
    plotfull_var(l_hist, [0,1,2], title, path)

def acc_var(l_hist, title=None, path=None):
    ''' plot accuracy with variance from a list of historys '''
    plt.ylim([0,1])
    plt.grid(True, which='major', linewidth=1, axis='y', alpha=1)
    plt.minorticks_on()
    plt.grid(True, which='minor', linewidth=0.8, axis='y', alpha=0.8)
    plotfull_var(l_hist, [3], title, path)

def l2_var(l_hist, title=None, path=None):
    '''plot l2 norm of gen model from a list of historys'''
    plotfull_var(l_hist, [4,5], title, path)

def gradsp_var(l_hist, title=None, path=None):
    ''' plot scalar product of gradients between 2 consecutive epochs
        from a list of historys
    '''
    plotfull_var(l_hist, [6,7], title, path)

# plotting all we have
def plot_metrics(l_hist, title=None, path=None):
    '''plot and save the different metrics from list of historys'''
    acc_var(l_hist, title, path)  
    loss_var(l_hist, title, path)
    l2_var(l_hist, title, path)
    gradsp_var(l_hist, title, path)