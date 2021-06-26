from django.core.management.base import BaseCommand, CommandError
from numpy.core.numeric import full

from tournesol.models import Comparison
from settings.settings import VIDEO_FIELDS

import numpy as np
import torch
from ml.management.commands.flower import get_flower

CRITERIAS = [   "reliability", "importance", "engaging", "pedagogy", 
                "layman_friendly", "diversity_inclusion", "backfire_risk", 
                "better_habits", "entertaining_relaxing"]

#CRITERIAS = ["reliability"]
EPOCHS = 10


def fetch_data():
    """
    Fetches the data from the Comparisons model

    Returns:
    - comparison_data: list of [contributor_id: int, video_id_1: int, video_id_2: int, criteria: str, score: float, weight: float]
    """
    print("fetch")
    comparison_data = [
        [comparison.user_id, comparison.video_1_id, comparison.video_2_id, criteria, getattr(comparison, criteria), getattr(comparison, f"{criteria}_weight")]
        for comparison in Comparison.objects.all() for criteria in VIDEO_FIELDS
        if hasattr(comparison, criteria)]
   # print(comparison_data)
    return comparison_data

def select_criteria(comparison_data, crit):
    ''' extracts data for this criteria where score is not None '''
    l_ratings = [comp for comp in comparison_data if (comp[3] == crit and comp[4] is not None)]
    return l_ratings

def rescale_score(score):
    ''' rescales from 0-100 to [-1,1] float '''
    return score / 50 - 1

def shape_data(l_ratings):
    ''' 
    l_notes : list of not None notations for one criteria 
    Returns : one array with 4 columns : contribID, ID1, ID2, score ([0,100]) 
    '''
    l_cleared = [rating[:3] + [rescale_score(rating[4])] for rating in l_ratings]
    arr = np.asarray(l_cleared)
    return arr

def sort_by_first(arr):
    ''' sort array lines by first element of lines '''
    order = np.argsort(arr,axis=0)[:,0]
    return arr[order,:]

def distribute_data(arr, gpu=False):
    ''' distributes data on nodes according to user IDs '''
    arr = sort_by_first(arr) # sorting by user IDs
    data_distrib = [[]]
    user_ids = [arr[0][0]] 
    id = arr[0][0]
    num_node =  0
    for comp in arr:
        if comp[0]==id:
            data_distrib[num_node].append(comp[1:])
        else: # new user id
            data_distrib.append([comp[1:]])
            id = comp[0]
            user_ids.append(id)
            num_node += 1
    for i, node in enumerate(data_distrib):
        data_distrib[i] = torch.tensor(node)
    return data_distrib, user_ids

def in_and_out(comparison_data, criteria):
    ''' trains and returns video scores'''
    one_crit = select_criteria(comparison_data, criteria)
    full_data = shape_data(one_crit)
    distributed, users_ids = distribute_data(full_data)
    flow = get_flower()
    flow.set_allnodes(distributed, users_ids)
    h = flow.train(EPOCHS, verb=2)
    glob, loc = flow.output_scores()
    return glob, loc, users_ids

def format_out_glob(glob, crit):
    ''' put data in list of global scores '''
    l_out = []
    ids, scores = glob
    for i in range(len(ids)):
        out = [ids[i].item(), crit, scores[i].item(), 0]
        l_out.append(out)
    return l_out

def format_out_loc(loc, users_ids, crit):
    ''' put data in list of local scores '''
    l_out = []
    vids, scores = loc
    for user_id, user_vids, user_scores in zip(users_ids, vids, scores):
        for i in range(len(user_vids)):
            out = [user_id, user_vids[i].item(), crit, user_scores[i].item(), 0]
            l_out.append(out)
    return l_out

def ml_run(comparison_data):
    """
    Uses data loaded

    Returns:
    - video_scores: list of [video_id: int, criteria_name: str, score: float, uncertainty: float]
    - contributor_rating_scores: list of [contributor_id: int, video_id: int, criteria_name: str, score: float, uncertainty: float]
    """ # not better to regroup contributors in same list or smthg ?
    video_scores, contributor_rating_scores = [], []
    for crit in CRITERIAS:
        print("\nPROCESSING", crit)
        glob, loc, users_ids = in_and_out(comparison_data, crit) # training
        # putting in required shape for output
        out_glob = format_out_glob(glob, crit) 
        out_loc = format_out_loc(loc, users_ids, crit) 
        video_scores += out_glob
        contributor_rating_scores += out_loc
    return video_scores, contributor_rating_scores

def save_data(video_scores, contributor_rating_scores):
    """
    Saves in the scores for Videos and ContributorRatings
    """
    pass

class Command(BaseCommand):
    help = 'Runs the ml'

    def handle(self, *args, **options):
        comparison_data = fetch_data()
        video_scores, contributor_rating_scores = ml_run(comparison_data)
        save_data(video_scores, contributor_rating_scores)

        print(len(video_scores))
        print(len(contributor_rating_scores))