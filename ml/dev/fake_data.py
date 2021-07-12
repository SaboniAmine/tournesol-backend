import random
import numpy as np
from math import exp, sinh
import scipy.stats as st


# ----------- fake data generation ---------------
def fake_glob_scores(nb_vid, scale=1):
    """ Creates fake global scores for test 
    
    nb_vid (int): number of videos "generated"
    scale (float): variance of generated global scores 

    Returns: 
        (float array): fake global scores
    """
    glob_scores = np.random.normal(scale=scale, size=nb_vid)
    return glob_scores

def fake_loc_scores(distribution, glob_scores, w):
    """ Creates fake local scores for test 
    
    distribution (int list): list of videos rated by the user
    glob_scores (float array): fake global scores
    w (float): nodes weight

    Returns: 
        (list of list of couples): (vid, local score) for each video
                                            of each node
    """
    all_idxs = range(len(glob_scores))
    b = 1/w # scale of laplace noise
    l_nodes = []
    for nb_vids in distribution: # for each node
        pick_idxs = random.sample(all_idxs, nb_vids) # videos rated by user
        noises = np.random.laplace(size=nb_vids, scale=b) # random noise
        node = [ (idx, glob_scores[idx] + noise)    for idx, noise 
                                                    in zip(pick_idxs, noises) 
                ] # list of (video id , video local score)
        l_nodes.append(node)
    return l_nodes

def rate_density(r, a, b):
    """ Returns density of r knowing a and b 
    
    r (float in [-1, 1]): comparison rate
    a (float): local score of video a
    b (float): local score of video b

    Returns:
        (float): density of r knowing a and b
    """
    t = a - b
    dens = t * exp(-r*t) / (2 * sinh(t))
    return dens
    
def get_rd_rate(a, b):
    """ Gives a random comparison score 
    
    a (float): local score of video a
    b (float): local score of video b
    
    Returns:
        (float): random comparison score
    """
    class my_pdf(st.rv_continuous):
        def _pdf(self, r):
            return rate_density(r, a, b)
    my_cv = my_pdf(a=-1, b=1, name='my_pdf')
    return my_cv.rvs()

def unscale_rating(r):
    """ Converts [-1,1] to [0, 100] """
    return (r + 1) * 50

def fake_comparisons(l_nodes, dens=0.5, crit="reliability"):
    """ 

    l_nodes (list of list of couples): (vid, local score) for each video
                                                            of each node
    crit (str): criteria of comparisons
    dens (float [0,1[): density of comparisons

    Returns:
        (list of lists): list of all comparisons
                    [   contributor_id: int, video_id_1: int, video_id_2: int, 
                        criteria: str, score: float, weight: float  ]
    """
    all_comps = []
    for uid, node in enumerate(l_nodes): # for each node
        nbvid = len(node)
        for vidx1, video in enumerate(node): # for each video
            nb_comp = int(dens * (nbvid - vidx1)) # number of comparisons
            following_videos = range(vidx1 + 1, nbvid) 
            pick_idxs = random.sample(following_videos, nb_comp)
            for vidx2 in pick_idxs:
                r = get_rd_rate(video[1], node[vidx2][1]) # get random r
                rate = unscale_rating(r)  # put to [0, 100]
                comp = [uid, video[0], node[vidx2][0], crit, rate, 0]
                all_comps.append(comp)
    return all_comps

def generate_data(nb_vid, nb_user, vids_per_user, dens=0.5):
    """ Generates fake input data for testing
    
    nb_vid (int): number of videos
    nb_user (int): number of users
    vids_per_user (int): number of videos rated by each user
    dens (float [0,1[): density of comparisons for each user

    Returns:
        (list of lists): list of all comparisons
            [   contributor_id: int, video_id_1: int, video_id_2: int, 
                criteria: "reliability", score: float, weight: float  ]
        (float array): fake global scores
        (list of list of couples): (vid, local score) for each video
                                                    of each node
    """
    distr = [vids_per_user] * nb_user
    glob = fake_glob_scores(nb_vid)
    print(nb_vid, 'global scores generated')
    loc = fake_loc_scores(distr, glob, w=1)
    print(vids_per_user, 'local scores generated per user')
    comp = fake_comparisons(loc, dens)
    print('comparisons generated')
    return comp, glob, loc
