
from os import EX_NOPERM
from django.core.management.base import BaseCommand, CommandError
from numpy.core.numeric import full

from tournesol.models import Comparison
from settings.settings import VIDEO_FIELDS

import numpy as np
import torch


from ml.management.commands.flower import get_flower
from ml.management.commands.utilities import rescale_rating, sort_by_first, one_hot_vids, get_all_vids
from ml.management.commands.utilities import reverse_idxs, disp_one_by_line, seedall, check_one, get_mask
from ml.management.commands.utilities import save_to_json, load_from_json, save_to_pickle, load_from_pickle
from ml.management.commands.utilities import expand_dic
"""
Machine Learning main python file

Organisation:
- Data is handled here
- ML model and decentralised structure are in "flower.py"
- some helpful small functions are in "utilities.py"

Notations:
- node = user : contributor
- vid = vID : video, video ID
- rating : rating provided by a contributor between 2 videos, in [0,100] or [-1,1]
- score : score of a video outputted by the algorithm, arround [-10, 10]
- glob, loc : global, local
- idx : index
- l_someting : list of someting
- arr : numpy array
- tens : torch tensor
- dic : dictionnary
- verb : verbosity level
- VARIABLE_NAME : global variable

Structure:
- fetch_data() provides data from the database
- ml_run() uses this data as input, trains via in_and_out()
     and returns video scores
- save_data() takes these scores and save them (TO DO)
- these 3 are called by Django at the end of this file

USAGE:
- define global variables EPOCHS and CRITERIAS
-- EPOCHS: number of training epochs
-- CRITERIAS: list of str (there is one training for each criteria)
- call ml_run(fetch_data()) if you just want the scores in python

"""
# global variables

EXPERIMENT_MODE = True
SAVE_PATH = "ml/models_weights"

if EXPERIMENT_MODE:
    CRITERIAS = ["reliability"]
else:
    CRITERIAS = [  "reliability", "importance", "engaging", "pedagogy", 
                    "layman_friendly", "diversity_inclusion", "backfire_risk", 
                    "better_habits", "entertaining_relaxing"]
    EPOCHS = 100





def fetch_data():
    """
    Fetches the data from the Comparisons model

    Returns:
    - comparison_data: list of [contributor_id: int, video_id_1: int, video_id_2: int, criteria: str, score: float, weight: float]
    """
    comparison_data = [
        [comparison.user_id, comparison.video_1_id, comparison.video_2_id, criteria, getattr(comparison, criteria), getattr(comparison, f"{criteria}_weight")]
        for comparison in Comparison.objects.all() for criteria in VIDEO_FIELDS
        if hasattr(comparison, criteria)]
    return comparison_data

def select_criteria(comparison_data, crit):
    ''' Extracts data for this criteria where score is not None 

    comparison_data: output of fetch_data()
    crit: str, name of criteria
        
    Returns: 
    - list of all ratings for this criteria
        (one element is [contributor_id: int, video_id_1: int, video_id_2: int, criteria: str (crit), score: float, weight: float])
    '''
    l_ratings = [comp for comp in comparison_data if (comp[3] == crit and comp[4] is not None)]
    return l_ratings

def shape_data(l_ratings):
    ''' 
    l_ratings : list of not None ratings for one criteria, all users

    Returns : one array with 4 columns : userID, vID1, vID2, rating ([-1,1]) 
    '''
    l_cleared = [rating[:3] + [rescale_rating(rating[4])] for rating in l_ratings]
    arr = np.asarray(l_cleared)
    return arr

def distribute_data(arr, gpu=False): # change to add user ID to tuple
    ''' 
    Distributes data on nodes according to user IDs for one criteria

    arr : np 2D array of all ratings for all users for one criteria
            (one line is [userID, vID1, vID2, score])

    Returns:
    - list of (user ID, vID1_batch, vID2_batch, rating_batch, single_vIDs_batch, mask) (1/user)
    - dictionnary of {vID: video idx}
    '''
    arr = sort_by_first(arr) # sorting by user IDs
    user_ids, first_of_each = np.unique(arr[:,0], return_index=True)
    first_of_each = list(first_of_each) # to be able to append
    first_of_each.append(len(arr)) # to have last index too
    vids = get_all_vids(arr)  # all unique video IDs
    vid_vidx = reverse_idxs(vids)
    nodes_dic = {}   # futur dictionnary of data for each user

    # for i in range(len(first_of_each) - 1):
    for i, id in enumerate(user_ids):
        node_arr = arr[first_of_each[i]: first_of_each[i+1], :]
        l1 = node_arr[:,1]
        l2 = node_arr[:,2]
        batchvids = get_all_vids(node_arr) # unique video IDs of node
        batch1 = one_hot_vids(vid_vidx, l1)
        batch2 = one_hot_vids(vid_vidx, l2)
        mask = get_mask(batch1, batch2)
        batchout = torch.FloatTensor(node_arr[:,3])
        nodes_dic[id] = (batch1, batch2, batchout, batchvids, mask)

    return nodes_dic, user_ids, vid_vidx

def distribute_data_from_save(arr, gpu=False):

    _, dic_old, _, _ = torch.load(SAVE_PATH) # loading previous data

    arr = sort_by_first(arr) # sorting by user IDs
    user_ids, first_of_each = np.unique(arr[:,0], return_index=True)
    first_of_each = list(first_of_each) # to be able to append
    first_of_each.append(len(arr)) # to have last index too
    vids = get_all_vids(arr)  # all unique video IDs
    vid_vidx = expand_dic(dic_old, vids) # updated dictionnary
    nodes_dic = {}    # futur list of data for each user

    #for i in range(len(first_of_each) - 1):
    for i, id in enumerate(user_ids):
        node_arr = arr[first_of_each[i]: first_of_each[i+1], :]
        l1 = node_arr[:,1]
        l2 = node_arr[:,2]
        batchvids = get_all_vids(node_arr) # unique video IDs of node
        batch1 = one_hot_vids(vid_vidx, l1)
        batch2 = one_hot_vids(vid_vidx, l2)
        mask = get_mask(batch1, batch2)
        batchout = torch.FloatTensor(node_arr[:,3])
        nodes_dic[id] = (batch1, batch2, batchout, batchvids, mask)

    return nodes_dic, user_ids, vid_vidx

def in_and_out(comparison_data, criteria, epochs, verb=2):
    ''' 
    Trains models and returns video scores for one criteria

    comparison_data: output of fetch_data()
    criteria: str, rating criteria
    
    Returns :   
    - (tensor of all vIDS , tensor of global video scores)
    - (list of tensor of local vIDs , list of tensors of local video scores)
    - list of users IDs in same order as second output
    '''
    one_crit = select_criteria(comparison_data, criteria)
    full_data = shape_data(one_crit)
    if RESUME and EXPERIMENT_MODE:
        nodes_dic, users_ids, dic = distribute_data_from_save(full_data)
        flow = get_flower(len(dic), dic, criteria) # dic: {video ID: video index}
        flow.load_and_update(nodes_dic, users_ids)
    else:
        nodes_dic, users_ids, dic = distribute_data(full_data)
        flow = get_flower(len(dic), dic, criteria) # dic: {video ID: video index}
        flow.set_allnodes(nodes_dic, users_ids)
    h = flow.train(epochs, verb=verb) 
    glob, loc = flow.output_scores()
    flow.save_models()
    if EXPERIMENT_MODE:
        flow.check() # some tests
        print("nb_nodes", flow.nb_nodes)
    return glob, loc, users_ids

def format_out_glob(glob, crit):
    ''' 
    Puts data in list of global scores (one criteria)
    
    glob: global scores in 2D tensor (1 line: [vID, score])
    crit: criteria
    
    Returns: 
    - list of [video_id: int, criteria_name: str, score: float, uncertainty: float]
    '''
    l_out = []
    ids, scores = glob
    for i in range(len(ids)):
        out = [int(ids[i]), crit, round(scores[i].item(), 2), 0] # uncertainty is 0 for now
        l_out.append(out)
    return l_out

def format_out_loc(loc, users_ids, crit):
    ''' 
    Puts data in list of local scores (one criteria)

    loc: 
    users_ids: list of user IDs in same order
    
    Returns : 
    - list of [contributor_id: int, video_id: int, criteria_name: str, score: float, uncertainty: float]
    '''
    l_out = []
    vids, scores = loc
    for user_id, user_vids, user_scores in zip(users_ids, vids, scores):
        for i in range(len(user_vids)):
            out = [int(user_id), int(user_vids[i].item()), 
                    crit, round(user_scores[i].item(), 2), 0] # uncertainty is 0 for now
            l_out.append(out)
    return l_out

def ml_run(comparison_data, epochs, verb=2):
    """ Runs the ml algorithm for all CRITERIAS (global variable)
    
    comparison_data: output of fetch_data()

    Returns:
    - video_scores: list of [video_id: int, criteria_name: str, score: float, uncertainty: float]
    - contributor_rating_scores: list of [contributor_id: int, video_id: int, criteria_name: str, score: float, uncertainty: float]
    """ # not better to regroup contributors in same list or smthg ?
    global_scores, local_scores = [], []
    for crit in CRITERIAS:
        print("\nPROCESSING", crit)
        glob, loc, users_ids = in_and_out(comparison_data, crit, epochs, verb) # training, see "flower.py"
        # putting in required shape for output
        out_glob = format_out_glob(glob, crit) 
        out_loc = format_out_loc(loc, users_ids, crit) 
        global_scores += out_glob
        local_scores += out_loc
    return global_scores, local_scores

def save_data(global_scores, local_scores):
    """
    Saves in the scores for Videos and ContributorRatings
    """
    pass


# ============= for experiments only ========= production code below this
TEST_DATA = [
                [0, 100, 101, "reliability", 100, 0],
                [1, 100, 101, "reliability", 100, 0],
                # [1, 100, 101, "reliability", 100, 0],
                # [0, 101, 110, "reliability", 0, 0],
                # [1, 102, 103, "reliability", 70, 0],
                # [2, 104, 105, "reliability", 50, 0],
                # [3, 106, 107, "reliability", 30, 0],
                # [4, 108, 109, "reliability", 30, 0],
                # [5, 208, 209, "reliability", 0, 0],
                # [8, 200, 201, "reliability", 0, 0],
                # [67, 200, 201, "reliability", 0, 0]
            ] #+ [[0, 555, 556, "reliability", 40, 0]] * 10 

NAME = ""
EPOCHSEXP = 60
TRAIN = True 
RESUME = True

class Command(BaseCommand):
    help = 'Runs the ml'
    def handle(self, *args, **options):
        if EXPERIMENT_MODE: 
            if TRAIN:
                seedall(2)
                comparison_data = fetch_data()
                global_scores, contributor_scores = ml_run(comparison_data[:10000], EPOCHSEXP, verb=1)
                save_to_json(global_scores, contributor_scores, NAME)
            else:
                global_scores, contributor_scores = load_from_json(NAME)
                
            disp_one_by_line(global_scores[:10])
            disp_one_by_line(contributor_scores[:10])
            check_one(5410, global_scores, contributor_scores)
            # check_one(5620, global_scores, contributor_scores)
            # check_one(5667, global_scores, contributor_scores)
            # check_one(7887, global_scores, contributor_scores)
            # for c in global_scores:
            #     if c[2]>3:
            #         print(c)
            print("global:", len(global_scores),"local:",  len(contributor_scores))

# =================== PRODUCTION ========================
        # just train all and predict if not experiment mode
        else: 
            RESUME = False
            comparison_data = fetch_data()
            global_scores, local_scores = ml_run(comparison_data, EPOCHS, verb=0)
            save_data(global_scores, local_scores)