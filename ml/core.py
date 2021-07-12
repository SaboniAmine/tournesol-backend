import os
import logging

from ml.licchavi import get_licchavi
from ml.handle_data import select_criteria, shape_data
from ml.handle_data import distribute_data, distribute_data_from_save
from ml.handle_data import format_out_loc, format_out_glob
from ml.dev.experiments import licch_stats, scores_stats

TOURNESOL_DEV = bool(int(os.environ.get("TOURNESOL_DEV", 0))) # dev mode

FOLDER_PATH = "ml/checkpoints/" 
FILENAME = "models_weights"
PATH = FOLDER_PATH + FILENAME
os.makedirs(FOLDER_PATH, exist_ok=True)

def shape_train_predict(comparison_data, crit, epochs, resume, verb=2):
    ''' Trains models and returns video scores for one criteria

    comparison_data: output of fetch_data()
    criteria: str, rating criteria
    resume: bool, resume previous processing
    Returns :   
    - (tensor of all vIDS , tensor of global video scores)
    - (list of tensor of local vIDs , list of tensors of local video scores)
    - list of users IDs in same order as second output
    '''
    one_crit = select_criteria(comparison_data, crit)
    full_data = shape_data(one_crit)
    fullpath = PATH + '_' + crit
    if resume:
        nodes_dic, users_ids, vid_vidx = distribute_data_from_save( full_data, 
                                                                    crit, 
                                                                    fullpath)
        licch = get_licchavi(len(vid_vidx), vid_vidx, crit) 
        licch.load_and_update(nodes_dic, users_ids, fullpath, verb)
    else:
        nodes_dic, users_ids, vid_vidx = distribute_data(full_data)
        licch = get_licchavi(len(vid_vidx), vid_vidx, crit)
        licch.set_allnodes(nodes_dic, users_ids, verb)
    h = licch.train(epochs, verb=verb) 
    glob, loc = licch.output_scores()
    licch.save_models(fullpath)
    if TOURNESOL_DEV: # some prints and plots
        licch_stats(licch)
        scores_stats(glob[1])
    return glob, loc, users_ids

def ml_run(comparison_data, epochs, criterias, resume, verb=2):
    """ Runs the ml algorithm for all CRITERIAS (global variable)
    
    comparison_data: output of fetch_data()

    Returns:
    - video_scores: list of [video_id: int, criteria_name: str, 
                                score: float, uncertainty: float]
    - contributor_rating_scores: list of 
    [   contributor_id: int, video_id: int, criteria_name: str, 
        score: float, uncertainty: float]
    """ # FIXME: not better to regroup contributors in same list or smthg ?
    glob_scores, loc_scores = [], []
    for criteria in criterias:
        logging.info("PROCESSING " + criteria)
        glob, loc, users_ids = shape_train_predict( comparison_data, 
                                                    criteria, 
                                                    epochs,
                                                    resume, 
                                                    verb) 
        # putting in required shape for output
        out_glob = format_out_glob(glob, criteria) 
        out_loc = format_out_loc(loc, users_ids, criteria) 
        glob_scores += out_glob
        loc_scores += out_loc
    return glob_scores, loc_scores