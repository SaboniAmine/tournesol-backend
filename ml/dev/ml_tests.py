import numpy as np
import torch
import pytest

from ml.data_utility import rescale_rating, get_all_vids, get_mask, reverse_idxs
from ml.data_utility import sort_by_first, expand_dic, expand_tens
from ml.handle_data import select_criteria, shape_data, distribute_data
from ml.losses import fbbt, hfbbt, fit_loss, s_loss, models_dist, model_norm
from ml.metrics import extract_grad, scalar_product
from ml.licchavi import get_licchavi, get_model
from ml.hyperparameters import get_defaults

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
    assert (output[len1 + 1:] == torch.zeros(len2) ).all()
    assert output.requires_grad

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

def test_models_dist():
    model1 = torch.tensor([1, 2, 4, 7])
    model2 = torch.tensor([3, -2, -5, 9.2])
    mask = torch.tensor([True, False, False, True])
    assert models_dist(model1, model2) == 17.2
    assert models_dist(model1, model2, mask=[mask]) == 4.2

def test_model_norm():
    model = torch.tensor([1, 2, -4.4, 7])
    assert model_norm(model) == 73.36

# --------- licchavi.py ------------
def test_get_model():
    model = get_model(6)
    assert (model == torch.zeros(6)).all()
    assert model.requires_grad

def test_get_licchavi():
    licch = get_licchavi(0, {}, "reliability")
    defaults = get_defaults()
    for key, param in defaults.items():
        assert getattr(licch, key) == param # check if default is applied

# -------- metrics.py --------------
def test_extract_grad():
    model = torch.ones(4, requires_grad=True)
    loss = model.sum() * 2
    loss.backward()
    for a, b in zip(extract_grad(model)[0], torch.ones(4) * 2):
        assert a == b

def test_scalar_product():
    l_grad1 = [ torch.ones(2), torch.ones(3)]
    l_grad2 = [ torch.zeros(2), torch.ones(3)]
    l_grad1[1][1] = 7
    output = scalar_product(l_grad1, l_grad2)
    assert output == 9

# ============= wider tests =============

def test_training_pipeline():
    """ checks that outputs of training have normal length """
    nb_vids, nb_users, vids_per_user = 5, 3, 5
    fake_data, _, _ = generate_data(    nb_vids, 
                                        nb_users, 
                                        vids_per_user,
                                        dens=0.999)
    # FIXME ml_run shouldnt save here for test
    glob_scores, contributor_scores = ml_run (  fake_data,
                                                epochs=1,
                                                criterias=["reliability"], 
                                                resume=False, 
                                                verb=-1)
    assert nb_vids <= len(glob_scores) <= vids_per_user
    assert len(contributor_scores) == nb_users * vids_per_user

# ======= scores quality tests =============
def id_score_assert(id, score, glob):
    if glob[0] == id:
        assert glob[3] == score

def test_simple_train():
    """ test coherency of results for few epochs and very light data """
    comparison_data = [                      
                        [1, 101, 102, "reliability", 100, 0],
                        [2, 100, 101, "largely_recommended", 100, 0],
                        [1, 104, 105, "reliability", 30, 0],
                        [99, 100, 101, "largely_recommended", 100, 0],
                        [2, 108, 107, "reliability", 10, 0],
                        [0, 100, 102, "reliability", 70, 0],
                        [0, 104, 105, "reliability", 70, 0],
                        [0, 109, 110, "reliability", 50, 0],
                        [2, 107, 108, "reliability", 10, 0],
                        [1, 100, 101, "reliability", 100, 0],
                        [3, 200, 201, "reliability", 85, 0],
                        ]
    # FIXME ml_run shouldnt save here for test
    glob_scores, loc_scores = ml_run (  comparison_data,
                                                epochs=2,
                                                criterias=["reliability"], 
                                                resume=False, 
                                                verb=-1)
    nb = [0, 0, 0, 0]
    for loc in loc_scores:
        assert loc[0] in [0, 1, 2, 3]
        nb[loc[0]] += 1 # counting local scores
    assert nb == [6, 5, 2, 2]
    for glob in glob_scores:
        id_score_assert(107, 0, glob)
        id_score_assert(108, 0, glob)
        id_score_assert(109, 0, glob)
        id_score_assert(110, 0, glob)
        if glob[0] == 102: # best rated video
            best = glob[3]
        if glob[0] == 100: # worst rated video
            worst = glob[3]
        if glob[0] == 200: # test symetric scores
            sym = glob[3]
    for glob in glob_scores:
        assert worst <= glob[3] <= best
        if glob[0] == 201:
            assert glob[3] == -sym # test symetric scores

def test_output_distribution():
    """ trains for more epochs with more data, 
    tests distribution at equilibrium
    """
    pass #TODO
              