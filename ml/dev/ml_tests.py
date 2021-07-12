import numpy as np
import torch
import pytest

from ml.data_utility import rescale_rating, get_all_vids, get_mask
from ml.data_utility import sort_by_first, reverse_idxs
from ml.data_utility import expand_dic, expand_tens


"""
Test module for ml

Main file is "ml_train.py"
"""

TEST_DATA = [
                [1, 100, 101, "reliability", 100, 0],
                [1, 100, 101, "reliability", 100, 0],
                [1, 100, 101, "reliability", 100, 0],
                [0, 100, 101, "reliability", 100, 0],
                [0, 100, 101, "reliability", 100, 0],
            ]

# @pytest.mark.unit
# def test_r():
#     assert False
def dic_inclusion(a, b):
    """ checks if a is included in 

    a (dictionnary)
    b (dictionnary)
    
    Returns:
        (bool): True if a included in b
    """
    return all([item in b.items() for item in a.items()])

# ---------- data_utility.py ----------------
def test_rescale_rating():
    assert rescale_rating(100) == 1
    assert rescale_rating(0) == -1

def test_get_all_vids():
    size = 50
    input = np.reshape(np.arange(4 * size), (size, 4))
    assert len(get_all_vids(input)) == 2 * size

def test_get_mask():
    input = torch.randint(2, (5,4), dtype=bool) # boolean tensor
    input2 = torch.zeros((5,4), dtype=bool)
    mask = get_mask(input, input2)
    assert mask[0].shape == torch.Size([4]) 

def test_sort_by_first():
    size = 50
    arr = np.reshape(np.array(range(2 * size, 0, -1)), (size, 2))
    sorted = sort_by_first(arr)
    assert (sorted == sort_by_first(sorted)).all() # second sort changes nothing
    assert isinstance(sorted, np.ndarray) # output is a numpy array

def test_reverse_idx():
    size = 20
    vids = np.arange(0, size, 2)
    vid_vidx = reverse_idxs(vids) # {vid: vidx} dic
    vids2 = np.zeros(size//2)
    for vid in vids:
        vids2[vid_vidx[vid]] = vid
    print(vids, vids2)
    assert (vids2 == vids).all() # output can reconstitute input
    assert isinstance(vid_vidx, dict)  # output is a dictionnary

def test_expand_tens():
    len1, len2 = 4, 2
    tens = torch.ones(len1)
    output = expand_tens(tens, len2)
    assert len(output) == len1 + len2 # good final length
    # expanded with zeros and with gradient needed
    assert ( output[len1 + 1:] == torch.zeros(len2, requires_grad=True) ).all()

def test_expand_dic():
    vid_vidx = {100:0, 200:2, 300:1}
    l_vid_new = [200, 500, 700]
    dic_new = expand_dic(vid_vidx, l_vid_new)
    assert dic_inclusion(vid_vidx, dic_new) # new includes old
    assert len(dic_new) == 5 # new has good length
    assert (500 in dic_new.keys()) and (700 in dic_new.keys()) # new updated










