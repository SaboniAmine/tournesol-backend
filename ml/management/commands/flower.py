
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from time import time

from ml.management.commands.utilities import extract_grad, sp, nb_params, models_dist
from ml.management.commands.utilities import model_norm, round_loss, node_local_loss, one_hot_vids
from ml.management.commands.utilities import expand_tens
"""
Machine Learning algorithm, used in "ml_train"

Organisation:
- ML model and decentralised structure are here
- Data is handled in "ml_train"
- some helpful small functions are in "utilities.py"


Notations:
- node = user : contributor
- vid = vID : video, video ID
- score : score of a video outputted by the algorithm, range?

- idx : index
- l_someting : list of someting
- arr : numpy array
- tens : torch tensor
- dic : dictionnary

Structure:
- Flower class is the structure designed to include
    a global model and one for each node
-- read Flower __init__ comments to better understand

USAGE:
- hardcode training hyperparameters in Flower __init__
- use get_flower() to get an empty Flower
- use Flower.set_allnodes() to populate nodes
- use Flower.train() to train the models
- use Flower.output_scores() to get the results

"""

# def get_classifier(nb_vids, gpu=False, zero_init=True):
#     ''' returns one layer model for one-hot entries '''
#     model = nn.Sequential(nn.Linear(nb_vids, 1, bias=False))
#     if zero_init:
#         with torch.no_grad():
#             for p in model.parameters():
#                 _ = p.zero_()
#     if gpu:
#         return model.cuda()
#     return model

def get_model(nb_vids, gpu=False, zero_init=True):
    model = torch.zeros(nb_vids, requires_grad=True)
    if gpu:
        return model.cuda()
    return model


# nodes organisation
class Flower():
    ''' Training structure including local models and general one 
        Allowing to add and remove nodes at will
        .pop
        .add_nodes
        .rem_nodes
        .train
        .display
        .check
    '''

    def __init__(self, nb_vids, dic, crit, gpu=False, **kwargs):
        ''' 
        nb_vids: number of different videos rated by at least one contributor for this criteria
        dic: dictionnary of {vID: idx}
        '''
        self.nb_params = nb_vids  # number of parameters of the model(= nb of videos)
        self.dic = dic # {video ID : video index (for that criteria)}
        self.gpu = gpu # boolean for gpu usage (not implemented yet)
        self.criteria = crit # criteria learnt by this Flower

        self.opt = optim.SGD
        self.lr_node = 1     # local learning rate (local scores)
        self.lr_gen = 0.1  # global learning rate (global scores)
        self.lr_s = 0.01     # local learning rate for s parameter
        self.gen_freq = 1  # generalisation frequency (>=1)
        self.w0 = 0.01      # regularisation strength
        self.w = 0.1    # default weight for a node

        self.get_model = get_model # neural network to use
        self.general_model = self.get_model(nb_vids, gpu)
        self.init_model = deepcopy(self.general_model) # saved for metrics
        self.last_grad = None
        self.opt_gen = self.opt([self.general_model], lr=self.lr_gen)
        self.pow_gen = (1,1)  # choice of norms for Licchavi loss 
        self.pow_reg = (2,1)  # (internal power, external power)


        self.nb_nodes = 0
        self.nodes = [] # list of tuples
        # (0:userID, 1:vID1_batch, 2:vID2_batch, 3:rating_batch, 4:single_vIDs_batch
        #   5: model, 6: s parameter, 7:optimizer, 8:weight, 9:age
        # )
        # self.size = nb_params(self.general_model) / 10_000
        self.history = ([], [], [], [], [], [], []) # all metrics recording (not totally up to date)
        # ("fit", "gen", "reg", "acc", "l2_dist", "l2_norm", "grad_sp", "grad_norm")
        
    # ------------ input and output --------------------
    def _get_default(self):
        ''' Returns default model + '''
        model_plus = (  torch.ones(1, requires_grad=True), 
                        self.get_model(self.nb_params, self.gpu),
                        0 #age
                        )
        return model_plus

    def _get_saved(self, loc_models_old, id, nb_new):
        ''' get saved parameters updated or default '''
        model = loc_models_old.get(id, self.get_default())
        if id in loc_models_old.keys():
            s, mod, age = model
            mod = expand_tens(mod, nb_new)
            model = (s, mod, age)
        return model

    def set_allnodes(self, data_dic, user_ids, verb=1):
        ''' Puts data in Flower and create a model for each node 
        
        data_distrib: data distributed by ml_train.distribute_data() 
                       ie list of (vID1_batch, vID2_batch, rating_batch, single_vIDs_batch)
        users_id: list/array of users IDs in same order 
        '''
        nb = len(data_dic)
        self.nb_nodes = nb
        self.users = user_ids
        self.nodes = { id:  [   *data, # 0 to 4: vID1, vID2, r, vIDs, mask
                                *self._get_default(), # 5 to 7: s, model, age
                                None,  # 8: optimizer
                                self.w, # 9: weight
                            ] for id, data in zip(user_ids, data_dic.values())}
        for node in self.nodes.values():
            node[8] = self.opt( [   {'params': node[6]}, 
                                    {'params': node[5], 'lr': self.lr_s},
                                        ], lr=self.lr_node)
        if verb:
            print("Total number of nodes : {}".format(self.nb_nodes))

    def output_scores(self):
        ''' Returns video scores both global and local
        
        Returns :   
        - (tensor of all vIDS , tensor of global video scores)
        - (list of tensor of local vIDs , list of tensors of local video scores)
        '''
        local_scores = []
        list_ids_batchs = []
        with torch.no_grad():
            glob_scores = self.general_model
            var = torch.var(glob_scores)
            mini, maxi = torch.min(glob_scores).item(),  torch.max(glob_scores).item()
            print("minimax:", mini,maxi)
            print("variance of global scores :", var.item())
            for node in self.nodes.values():
                input = one_hot_vids(self.dic, node[3])
                output = torch.matmul(input, node[6]) 
                local_scores.append(output)
                list_ids_batchs.append(node[3])
            vids_batch = list(self.dic.keys())
        return (vids_batch, glob_scores), (list_ids_batchs, local_scores)

    def save_models(self):
        ''' saves global and local weights, detached (no gradients) '''
        local_data = {id:  (node[5],            # s
                            node[6].detach(),   # model
                            node[7]            # age
                        ) for id, node in self.nodes.items()}
        saved_data = (  self.criteria,
                        self.dic,
                        self.general_model.detach(), 
                        local_data
                        )
        torch.save(saved_data, "ml/models_weights")

    def load_and_update(self, data_dic, user_ids, verb=1):
        ''' loads weights and expands them as required 
        nb_new: number of new videos
        '''
        self.criteria, dic_old, gen_model_old, loc_models_old = torch.load("ml/models_weights")
        nb_new = self.nb_params - len(dic_old) # number of new videos
        self.general_model = expand_tens(gen_model_old, nb_new) # initialize scores for new videos
        self.users = user_ids
        nbn = len(user_ids)
        self.nb_nodes = nbn
        
        self.nodes = { id: [    *data, # 0 to 4
                                *self._get_saved(loc_models_old, id, nb_new), # 5 to 7
                                None, # 8: optimizer
                                self.w # 9: weight
                            ] for id, data in zip(user_ids, data_dic.values())}                   
        for node in self.nodes.values(): # set optimizers
            node[8] = self.opt( [   {'params': node[6]}, 
                                    {'params': node[5], 'lr': self.lr_s},
                                        ], lr=self.lr_node)
        if verb:
            print("Total number of nodes : {}".format(self.nb_nodes))

    # ---------- methods for training ------------
    def _set_lr(self):
        ''' sets learning rates of optimizers according to Flower setting '''
        for node in self.nodes.values(): 
            node[8].param_groups[0]['lr'] = self.lr_node
        self.opt_gen.param_groups[0]['lr'] = self.lr_gen

    def _zero_opt(self):
        ''' resets gradients of all models '''
        for node in self.nodes.values():
            node[8].zero_grad()      
        self.opt_gen.zero_grad()

    def _update_hist(self, epoch, fit, gen, reg, verb=1):
        ''' updates history '''
        self.history[0].append(round_loss(fit))
        self.history[1].append(round_loss(gen))
        self.history[2].append(round_loss(reg))

        dist = models_dist(self.init_model, self.general_model, pow=(2,0.5)) 
        norm = model_norm(self.general_model, pow=(2,0.5))
        self.history[3].append(round_loss(dist, 1))
        self.history[4].append(round_loss(norm, 1))
        grad_gen = extract_grad(self.general_model)
        if epoch > 1: # no last model for first epoch
            scal_grad = sp(self.last_grad, grad_gen)
            self.history[5].append(scal_grad)
        else:
            self.history[5].append(0) # default value for first epoch
        self.last_grad = deepcopy(extract_grad(self.general_model)) 
        grad_norm = sp(grad_gen, grad_gen)  # use sqrt ?
        self.history[6].append(grad_norm)

    def _old(self, years):
        ''' increments age of nodes (during training) '''
        for node in self.nodes.values():
            node[7] += years

    def _counters(self, c_gen, c_fit):
        ''' updates internal training counters '''
        fit_step = (c_fit >= c_gen) 
        if fit_step:
            c_gen += self.gen_freq
        else:
            c_fit += 1 
        return fit_step, c_gen, c_fit

    def _do_step(self, fit_step):
        ''' step for appropriate optimizer(s) '''
        if fit_step:       # updating local or global alternatively
            for node in self.nodes.values(): 
                node[8].step()      
        else:
            self.opt_gen.step()  

    def _print_losses(self, tot, fit, gen, reg):
        ''' prints losses '''
        print("total loss : ", tot) 
        print("fitting : ", round_loss(fit, 2),
                ', generalisation : ', round_loss(gen, 2),
                ', regularisation : ', round_loss(reg, 2))

    def _rectify_s(self):
        ''' ensures that no s went under 0 '''
        limit = 0.01
        with torch.no_grad():
            for node in self.nodes.values():
                if node[5] < limit:
                    node[5][0] = limit

    # ====================  TRAINING ================== 

    def train(self, nb_epochs=None, verb=1):   
        ''' training loop '''
        nb_epochs = 2 if nb_epochs is None else nb_epochs
        time_train = time()
        self._set_lr()

        # initialisation to avoid undefined variables at epoch 1
        loss, fit_loss, gen_loss, reg_loss = 0, 0, 0, 0
        c_fit, c_gen = 0, 0

        const = 10  # just for visualisation (remove later)
        fit_scale = const 
        gen_scale = const  # node weights are used in addition
        reg_scale = const * self.w0 

        reg_loss = reg_scale * model_norm(self.general_model, self.pow_reg)  

        # training loop 
        nb_steps = self.gen_freq + 1
        for epoch in range(1, nb_epochs + 1):
            if verb: print("\nepoch {}/{}".format(epoch, nb_epochs))
            time_ep = time()

            for step in range(1, nb_steps + 1):
                fit_step, c_gen, c_fit = self._counters(c_gen, c_fit)
                if verb >= 2: 
                    txt = "(fit)" if fit_step else "(gen)" 
                    print("step :", step, '/', nb_steps, txt)
                self._zero_opt() # resetting gradients


                #----------------    Licchavi loss  -------------------------
                 # only first 2 terms of loss updated
                if fit_step:
                    
                    #self._rectify_s()  # to prevent s from diverging (bruteforce)
                    fit_loss, gen_loss = 0, 0
                    for node in self.nodes.values():  
                        fit_loss += node_local_loss(node[6],  # model
                                                    node[5],  # s
                                                    node[0],  # id_batch1
                                                    node[1],  # id_batch2
                                                    node[2])  # r_batch
                        g = models_dist(node[6], # model
                                        self.general_model, 
                                        self.pow_gen, 
                                        node[4] # mask
                                        #None
                                        ) 
                        gen_loss +=  node[9] * g  # node weight  * generalisation term
                    fit_loss *= fit_scale
                    gen_loss *= gen_scale
                    loss = fit_loss + gen_loss 
                          
                # only last 2 terms of loss updated 
                else:        
                    gen_loss, reg_loss = 0, 0
                    for node in self.nodes.values():   
                        g = models_dist(node[6], # model
                                        self.general_model, 
                                        self.pow_gen,
                                        node[4] # mask
                                        #None
                                        )
                        gen_loss += node[9] * g  # node weight  * generalisation term
                    reg_loss = model_norm(self.general_model, self.pow_reg) 
                    gen_loss *= gen_scale
                    reg_loss *= reg_scale       
                    loss = gen_loss + reg_loss

                if verb >= 2:
                    total_out = round_loss(fit_loss + gen_loss + reg_loss)
                    self._print_losses(total_out, fit_loss, gen_loss, reg_loss)
                    
                # Gradient descent 
                loss.backward() 
                self._do_step(fit_step)   
 
            if verb: print("epoch time :", round(time() - time_ep, 2)) 
            self._update_hist(epoch, fit_loss, gen_loss, reg_loss, verb)
            self._old(1)  # aging all nodes of 1 epoch
             
        # ----------------- end of training -------------------------------  
        print("training time :", round(time() - time_train, 2)) 
        return self.history # self.train() returns lists of metrics

    # ------------ to check for problems --------------------------
    def check(self):
        ''' perform some tests on internal parameters adequation '''
        # population check
        b1 =  (self.nb_nodes == len(self.nodes))
        # history check
        b2 = True
        for l in self.history:
            b2 = b2 and (len(l) == len(self.history[0]))
        print(b1, b2, self.nb_nodes ,len(self.nodes) , len(self.dic))
        if (b1 and b2):
            print("No Problem")
        else:
            print("Coherency problem in Flower object ")


def get_flower(nb_vids, dic, crit, gpu=False, **kwargs):
    ''' Returns a Flower (ml decentralized structure)

    nb_vids: number of different videos rated by at least one contributor for this criteria
    dic: dictionnary of {vID: idx}
    crit: criteria of users ratingd
    '''
    return Flower(nb_vids, dic, crit, gpu=gpu, **kwargs)

