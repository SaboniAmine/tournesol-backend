import numpy as np
import torch
import pytest

from ml.data_utility import rescale_rating, get_all_vids, get_mask
from ml.data_utility import sort_by_first, reverse_idxs
from ml.data_utility import expand_dic, expand_tens
from ml.handle_data import select_criteria, shape_data, distribute_data
from ml.losses import fbbt, hfbbt, fit_loss, s_loss

from ml.dev.fake_data import generate_data
from ml.core import ml_run


# """
# Test module for ml

# Main file is "ml_train.py"
# """

TEST_DATA = [
                [1, 100, 101, "reliability", 100, 0],
                [1, 101, 102, "reliability", 100, 0],
                [1, 104, 105, "reliability", 100, 0],
                [0, 100, 101, "reliability", 100, 0],
                [0, 100, 101, "largely_recommended", 100, 0],
            ]

# #@pytest.mark.unit


def dic_inclusion(a, b):
    """ checks if a is included in 

    a (dictionnary)
    b (dictionnary)
    
    Returns:
        (bool): True if a included in b
    """
    return all([item in b.items() for item in a.items()])

# ========== unit tests ===============

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

# -------- handle_data.py -------------
def test_select_criteria():
    comparison_data = TEST_DATA
    crit = "reliability"
    output = select_criteria(comparison_data, crit)
    for comp in output:
        assert len(comp) == 6 # len of each element of output list
    assert isinstance(output, list)
    assert len(output) == 4 # number of comparisons extracted

def test_shape_data():
    l_ratings = [   [0, 100, 101, "reliability", 100, 0],
                    [0, 100, 101, "reliability", 50, 0],
                    [0, 100, 101, "reliability", 0, 0],
                ]
    output = shape_data(l_ratings)
    assert isinstance(output, np.ndarray)
    assert output.shape == (3, 4)   # output shape
    assert np.max(abs(output[:,3])) <= 1 # range of scores

def test_distribute_data():
    arr = np.array([[0, 100, 101, 1],
                    [1, 100, 101, -1],
                    [0, 101, 102, 0],
                    ] )
    arr[1][0] = 3 # comparison 1 is performed by user or id 3
    nodes_dic, user_ids, vid_vidx = distribute_data(arr)
    assert len(nodes_dic) == 2 # number of nodes
    assert len(nodes_dic[0][0]) == 2 # number of comparisons for user 0
    assert len(nodes_dic[0][0][0]) == 3 # total number of videos
    assert len(user_ids) == len(nodes_dic) # number of users
    assert len(vid_vidx) == len(nodes_dic[0][0][0]) # total number of videos

# ------------ losses.py ---------------------
def test_fbbt_hfbbt():
    l_t = [-2, -0.5, 0.001, 0.1, 0.3, 10, 50]
    l_r = [-1, -0.8, -0.754, -0.2, -0.002, 0, 0.3, 0.564, 1]
    for tt in l_t:
        for rr in l_r:
            t = torch.tensor(tt)
            r = torch.tensor(rr)
            assert abs(fbbt(t, r) - hfbbt(t,r)) <= 0.001

def test_fit_loss():
    l_s = [ 0.4, 0.5, 0.67, 0.88, 0.1, 1.2]
    l_ya = [ 0.5, -0.2, 1.3, 3.4, -0.1, 0]
    l_yb = [ 0.8, -1.2, -1.3, 0.34, 1.1, 0.5]
    l_r = [0.2, 0.3, 0.4, -0.89, 0, 1]
    results = [0.6715, 0.8845, 1.8526, -0.699, 0.6955, 0.1524]
    for ss, yya, yyb, rr, res in zip(l_s, l_ya, l_yb, l_r, results):
        s = torch.tensor(ss)
        ya = torch.tensor(yya)
        yb = torch.tensor(yyb)
        r = torch.tensor(rr)
        output = fit_loss(s, ya, yb, r).item()
        assert abs(output - res) <= 0.001

def test_s_loss():
    l_s = [ 0.4, 0.5, 0.67, 0.88, 0.1, 1.2]
    results = [0.9963, 0.8181, 0.6249, 0.515, 2.3076, 0.5377]
    for ss, res in zip(l_s, results):
        s = torch.tensor(ss)
        output = s_loss(s).item()
        assert abs(output - res) <= 0.001

# ============= wider tests =============

def test_training_pipeline():
    """ checks that outputs of training have normal length """
    nb_vids, nb_users, vids_per_user = 5, 3, 5
    fake_data, _, _ = generate_data(    nb_vids, 
                                        nb_users, 
                                        vids_per_user,
                                        dens=0.999)
    glob_scores, contributor_scores = ml_run (  fake_data,
                                                1,
                                                ["reliability"], 
                                                resume=False, 
                                                verb=-1)
    assert nb_vids <= len(glob_scores) <= vids_per_user
    assert len(contributor_scores) == nb_users * vids_per_user
              