import numpy as np
import torch

"""
Test module called in dev mode only (in "experiments.py")

Main file is "ml_train.py"
"""

TEST_DATA = [
                [1, 100, 101, "reliability", 100, 0],
                [1, 100, 101, "reliability", 100, 0],
                [1, 100, 101, "reliability", 100, 0],
                [0, 100, 101, "reliability", 100, 0],
                [0, 100, 101, "reliability", 100, 0],
            ]

def run_unittests():
    """ runs some tests """

    # data_utility.py
    from ..data_utility import rescale_rating, get_all_vids, get_mask
    assert rescale_rating(100) == 1
    assert rescale_rating(0) == -1

    size = 50
    input = np.reshape(np.arange(4 * size), (size, 4))
    assert len(get_all_vids(input)) == 2 * size

    # get_mask
    input = torch.randint(2, (5,4), dtype=bool) # boolean tensor
    input2 = torch.zeros((5,4), dtype=bool)
    mask = get_mask(input, input2)
    assert mask[0].shape == torch.Size([4]), "get_mask"

    # sort_by_first
    from ..data_utility import sort_by_first, reverse_idxs
    arr = np.reshape(np.array(range(2 * size, 0, -1)), (size, 2))
    sorted = sort_by_first(arr)
    assert (sorted == sort_by_first(sorted)).all()
    assert isinstance(sorted, np.ndarray)

    # reverse_idxs
    size = 20
    vids = np.arange(0, size, 2)
    vid_vidx = reverse_idxs(vids) # {vid: vidx} dic
    vids2 = np.zeros(size//2)
    for vid in vids:
        vids2[vid_vidx[vid]] = vid
    print(vids, vids2)
    assert (vids2 == vids).all()
    assert isinstance(vid_vidx, dict) 
    #assert False, "ALL GOOD"

    # expand_tens
    from ..data_utility import expand_dic, expand_tens
    











