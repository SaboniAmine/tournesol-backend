from os import makedirs
import torch

from ..data_utility import save_to_json, load_from_json
from .plots import plot_metrics, plot_density
from .visualisation import seedall, check_one, disp_one_by_line
from .fake_data import generate_data
from .ml_tests import run_unittests

"""
Not used in production, for testing only
Module called from "ml_train.py" only if env var TOURNESOL_DEV is True

Used to perform some tests on ml algorithm (custom data, plots, ...)
"""

PATH_PLOTS = "ml/plots/"
makedirs(PATH_PLOTS, exist_ok=True)

CRITERIAS = ["reliability"]
TEST_DATA = [
                [1, 100, 101, "reliability", 100, 0],
                [1, 100, 101, "reliability", 100, 0],
                [1, 100, 101, "reliability", 100, 0],
                [0, 100, 101, "reliability", 100, 0],
                [0, 100, 101, "reliability", 100, 0],
                

                # [1, 100, 101, "reliability", 100, 0],
                # [2, 100, 101, "reliability", 100, 0],
                # [3, 100, 101, "reliability", 100, 0],
                # [4, 100, 101, "reliability", 100, 0],
                # [1, 100, 101, "importance", 100, 0],
                # [1, 100, 101, "reliability", 100, 0],
                # [1, 102, 103, "reliability", 70, 0],
                # [2, 104, 105, "reliability", 50, 0],
                # [3, 106, 107, "reliability", 30, 0],
                # [4, 108, 109, "reliability", 30, 0],
                # [67, 200, 201, "reliability", 0, 0]
            ] #+ [[0, 555, 556, "reliability", 40, 0]] * 10 


 

# nb_vids, nb_users, vids_per_user

NAME = ""
EPOCHS = 2
TRAIN = True 
RESUME = False

def run_experiment(comparison_data):
    """ trains and outputs some stats """
    if TRAIN:
        from ..management.commands.ml_train import ml_run
        seedall(4)
        run_unittests()
        #fake_data, glob_fake, loc_fake = generate_data(5, 3, 5, dens=0.999)
        glob_scores, contributor_scores = ml_run(comparison_data[:10000],
                                                    EPOCHS,
                                                    CRITERIAS, 
                                                    RESUME,
                                                    verb=1)
        save_to_json(glob_scores, contributor_scores, NAME)
    else:
        glob_scores, contributor_scores = load_from_json(NAME)
    for c in comparison_data:
        if c[3]=="largely_recommended":
            print(c)
    disp_one_by_line(glob_scores[:10])
    # disp_one_by_line(GLOB[:10])
    disp_one_by_line(contributor_scores[:10])
    # disp_one_by_line(LOC[0][:10])
    # check_one(6598, glob_scores, contributor_scores)
    # check_one(6865, glob_scores, contributor_scores)
    # check_one(7844, glob_scores, contributor_scores)
    # check_one(7928, glob_scores, contributor_scores)
    print("glob:", len(glob_scores), "local:",  len(contributor_scores))

def licch_stats(licch):
    ''' gives some statistics about Licchavi object '''
    licch.check() # some tests
    h = licch.history
    print("nb_nodes", licch.nb_nodes)
    licch.stat_s()  # print stats on s parameters
    with torch.no_grad():
        gen_s = licch.all_nodes("s")
        l_s = [s.item() for s in gen_s]
        plot_density(   l_s, 
                        "s parameters", 
                        PATH_PLOTS,
                        "s_params.png")
    plot_metrics([h], path=PATH_PLOTS)
    # print("uncertainty", licch.uncert)

def scores_stats(glob_scores):
    ''' gives statistics on global scores
    
    glob_scores: torch tensor of global scores
    '''
    var = torch.var(glob_scores)
    mini, maxi = (torch.min(glob_scores).item(),  
                torch.max(glob_scores).item() )
    print("minimax:", mini,maxi)
    print("variance of global scores :", var.item())
    with torch.no_grad():
        plot_density(glob_scores, "Global scores", PATH_PLOTS, "scores.png")
