import torch
import numpy as np
import json
import pickle 


# to handle data (used in ml_train.py)
def rescale_rating(rating):
    ''' rescales from 0-100 to [-1,1] float '''
    return rating / 50 - 1

def get_all_vids(arr):
    ''' get all unique vIDs for one criteria (all users) '''
    return np.unique(arr[:,1:3])

def get_mask(batch1, batch2):
    ''' get mask '''
    batch = batch1 + batch2
    to = batch.sum(axis=0, dtype=bool)
    return [to]

def sort_by_first(arr):
    ''' sorts 2D array lines by first element of lines '''
    order = np.argsort(arr,axis=0)[:,0]
    return arr[order,:]

def one_hot_vid(dic, vid):
    ''' One-hot inputs for neural network
    
    dic: dictionnary of {vID: idx}
    vid: vID

    Returns: 1D boolesn tensor with 0s and 1 only for video index
    '''
    tens = torch.zeros(len(dic), dtype=bool)
    tens[dic[vid]] = True
    return tens

def one_hot_vids(dic, l_vid):
    ''' One-hot inputs for neural network, list to batch
    
    dic: dictionnary of {vID: idx}
    vid: list of vID

    Returns: 2D bollean tensor with one line being 0s and 1 only for video index
    '''
    batch = torch.zeros(len(l_vid), len(dic), dtype=bool)
    for idx, vid in enumerate(l_vid):
        batch[idx][dic[vid]] = True
    return batch

def reverse_idxs(vids):
    ''' Returns dictionnary of {vid: idx} '''
    vid_vidx = {}
    for idx, vid in enumerate(vids):
        vid_vidx[vid] = idx
    return vid_vidx 

# used for updating models after loading
def expand_tens(tens, nb_new):
    ''' Expands a tensor to include scores for new videos
    
    tens: a detached tensor 

    Returns:
    - expanded tensor requiring gradients
    '''
    full = torch.cat([tens, torch.zeros(nb_new)])
    full.requires_grad=True
    return full

def expand_dic(dic, l_vid_new):
    ''' Expands a dictionnary to include new videos IDs

    dic: dictionnary of {video ID: video idx}
    l_vid_new: int list of video ID
    
    Returns:
    - dictionnary of {video ID: video idx} updated (bigger)
    '''
    idx = len(dic)
    for vid_new in l_vid_new:
        if vid_new not in dic:
            dic[vid_new] = idx
            idx += 1
    return dic

# save and load data
def save_to_json(global_scores, local_scores, suff=""):
    ''' saves scores in json files '''
    with open("global_scores{}.json".format(suff), 'w') as f:
        json.dump(global_scores, f, indent=1) 
    with open("local_scores{}.json".format(suff), 'w') as f:
        json.dump(local_scores, f, indent=1) 

def load_from_json(suff=""):
    ''' loads previously saved data '''
    with open("global_scores{}.json".format(suff), 'r') as f:
        global_scores = json.load(f)
    with open("local_scores{}.json".format(suff), 'r') as f:
        local_scores = json.load(f)
    return global_scores, local_scores

def save_to_pickle(obj, name="pickle"):
    ''' save python object to pickle file '''
    filename = '{}.p'.format(name)
    with open(filename, 'wb') as filehandler:
        pickle.dump(obj, filehandler)

def load_from_pickle(name="pickle"):
    filename = '{}.p'.format(name)
    with open(filename, 'rb') as filehandler:
        obj = pickle.load(filehandler)
    return obj