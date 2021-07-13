from torch.autograd.functional import hessian
import torch
from copy import deepcopy

from .losses import round_loss, loss_fit_s_gen

"""
Metrics used for training monitoring in "licchavi.py"

Main file "ml_train.py"
"""

# metrics on models
def extract_grad(model):
    ''' returns list of gradients of a model 
    
    model (float tensor): torch tensor with gradients

    Returns:
        (float tensor list): list of gradients of the model
    '''
    l_grad =  [p.grad for p in [model]]
    return l_grad

def scalar_product(l_grad1, l_grad2):
    ''' scalar product of 2 lists of gradients 
    
    l_grad1 (float tensor list): list of gradients of a model
    l_grad2 (float tensor list): list of gradients of a model

    Returns:
        (float): scalar product of the gradients
    '''
    s = 0
    for g1, g2 in zip(l_grad1, l_grad2):
        s += (g1 * g2).sum()
    return round_loss(s, 4)

def get_loc_models(nodes):
    ''' Returns a generator of all local models 
    
    nodes (Node dictionnary): dictionnary of all nodes

    Returns:
        (generator): generator of nodes' models
    '''
    for node in nodes.values():
        yield node.model

def replace_coordinate(tens, score, idx):
    """ Replaces one coordinate of the tensor

    Args:
        tens (float tensor): local model
        score (scalar tensor): score to put in tens
        idx (int): idx of score to replace

    Returns:
        (float tensor): same tensor as input but backward pointing to -score
    """
    size = len(tens)
    left, _, right = torch.split(tens, [idx, 1, size - idx - 1]) 
    new = torch.cat([left, score, right])
    return new

# to compute uncertainty
def get_hessian_fun(nodes, general_model, fit_scale, gen_scale, pow_gen,
                          id_node, vidx):
    """ Gives loss in function of local model for hessian computation 
    
    Args:
        nodes (Node dictionnary): dictionnary of all nodes
        general_model (float tensor): general model
        fit_scale (float): importance of the local loss
        gen_scale float): importance of the generalisation loss
        pow_gen (float, float): distance used for generalisation
        id_node (int): id of user
        vidx (int): index of video, ie index of parameter

    Returns:
        (scalar tensor -> float) function giving loss according to one parameter 
    """
    def get_loss(score):
        """ Used to compute its second derivative to get uncertainty
        
        input (float scalar tensor): one score

        Returns:
            (float scalar tensor): partial loss
        """

        new_model = replace_coordinate(nodes[id_node].model, score, vidx)
        nodes[id_node].model = new_model
        fit_loss, s_loss, gen_loss =  loss_fit_s_gen(nodes, general_model, fit_scale, 
                                    gen_scale, pow_gen)
        return fit_loss + s_loss + gen_loss
    return get_loss

def get_uncertainty(nodes, general_model, fit_scale, 
                        gen_scale, pow_gen, vid_vidx):
    """ Returns uncertainty for all local scores (list of list of int) 
    
    Args:
        nodes (Node dictionnary): dictionnary of all nodes
        general_model (float tensor): general model
        fit_scale (float): importance of the local loss
        gen_scale float): importance of the generalisation loss
        pow_gen (float, float): distance used for generalisation
        vid_vidx (dictionnary): {video ID: video index}

    Returns:
        (list of list of float): uncertainty for all local scores
    """
    l_uncert = []
    for uid, node in nodes.items(): # for all nodes
        local_uncerts = []
        for vid in node.vids:  # for all videos of the node
            vidx = vid_vidx[vid]  # video index
            score = node.model[vidx:vidx+1].detach()
            score = deepcopy(score)
            fun = get_hessian_fun(nodes, general_model, fit_scale,
                                gen_scale, pow_gen, uid, vidx)
            deriv2 = hessian(fun, score).item()
            uncert = deriv2**(-0.5)
            local_uncerts.append(uncert)
        l_uncert.append(local_uncerts)
    return l_uncert





