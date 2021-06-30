
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from time import time

from ml.management.commands.utilities import extract_grad, sp, nb_params, models_dist
from ml.management.commands.utilities import model_norm, round_loss, node_local_loss, one_hot_vids

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


def get_classifier(nb_vids, gpu=False, zero_init=True):
    ''' returns one layer model for one-hot entries '''
    model = nn.Sequential(nn.Linear(nb_vids, 1, bias=False))
    if zero_init:
        with torch.no_grad():
            for p in model.parameters():
                _ = p.zero_()
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

    def __init__(self, nb_vids, dic, gpu=False, **kwargs):
        ''' 
        nb_vids: number of different videos rated by at least one contributor for this criteria
        dic: dictionnary of {vID: idx}
        '''
        self.nb_params = nb_vids  # number of parameters of the model(= nb of videos)
        self.dic = dic # {video ID : video index (for that criteria)}
        self.gpu = gpu # boolean for gpu usage (not implemented yet)

        self.opt = optim.SGD
        self.lr_node = 1     # local learning rate (local scores)
        self.lr_gen = 0.02  # global learning rate (global scores)
        self.lr_s = 0.01     # local learning rate for s parameter
        self.gen_freq = 1  # generalisation frequency (>=1)
        self.w0 = 0.01      # regularisation strength
        self.w = 0.1     # default weight for a node

        self.get_classifier = get_classifier # neural network to use
        self.general_model = self.get_classifier(nb_vids, gpu)
        self.init_model = deepcopy(self.general_model) # saved for metrics
        self.last_grad = None
        self.opt_gen = self.opt(self.general_model.parameters(), lr=self.lr_gen)
        self.pow_gen = (1,1)  # choice of norms for Licchavi loss 
        self.pow_reg = (2,1)  # (internal power, external power)
        self.data = []  # list of nodes, one being (vID1_batch, vID2_batch, rating_batch, single_vIDs_batch)
        self.models = []  # one model for each node
        self.opt_nodes = [] # one optimizer for each model (1/node)
        self.s_nodes = []   # s parameter for each node (represents notation style)
        self.age = []          # number of epochs each node has been trained
        self.user_ids = []     # id of each node (user IDs)
        self.weights = []    # weight of each node
        self.nb_nodes = 0

        self.size = nb_params(self.general_model) / 10_000
        self.history = ([], [], [], [], [], [], []) # all metrics recording (not totally up to date)
        # self.h_legend = ("fit", "gen", "reg", "acc", "l2_dist", "l2_norm", "grad_sp", "grad_norm")
        
    # ------------ input and output --------------------
    def set_allnodes(self, data_distrib, user_ids, verb=1):
        ''' Puts data in Flower and create a model for each node 
        
        data_distrib: data distributed by ml_train.distribute_data() 
                       ie list of (vID1_batch, vID2_batch, rating_batch, single_vIDs_batch)
        users_id: list/array of users IDs in same order 
        '''
        self.data = data_distrib
        self.user_ids = user_ids
        nb = len(self.data)

        
        self.weights = [self.w] * nb
        self.age = [0] * nb
        self.nb_nodes = nb
        self.models = [self.get_classifier(self.nb_params, self.gpu) for i in range(nb)]
        self.s_nodes = [torch.ones(1, requires_grad=True) for n in range(nb)] # s is initialized at 1
        self.opt_nodes = [self.opt( [
                                    {'params': self.models[n].parameters()}, 
                                    {'params': self.s_nodes[n], 'lr': self.lr_s},
                                    ], lr=self.lr_node
                                    ) for n in range(nb)]
                            
        
        if verb:
            print("Total number of nodes : {}".format(self.nb_nodes))

    def output_scores(self):
        ''' Returns video scores both global and local
        
        Returns :   
        - (tensor of all vIDS , tensor of global video scores)
        - (list of tensor of local vIDs , list of tensors of local video scores)
        '''
        mean_choice = 0 # where to center scores
        local_scores = []
        list_ids_batchs = []
        with torch.no_grad():
            for p in self.general_model.parameters():  # only one iteration   
                glob_scores = p[0]
            m = torch.mean(glob_scores) # mean of scores to unbias
            var = torch.var(glob_scores)
            mini, maxi = torch.min(glob_scores).item(),  torch.max(glob_scores).item()
            print("minimax:", mini,maxi)
            print("variance of global scores :", var.item())
            glob_scores2 = glob_scores #+ mean_choice - m # set mean 
            for n, node in enumerate(self.data):
                input = one_hot_vids(self.dic, node[3])
                output = self.models[n](input) #+ mean_choice - m
                local_scores.append(output)
                list_ids_batchs.append(node[3])
            vids_batch = list(self.dic.keys())
        return (vids_batch, glob_scores2), (list_ids_batchs, local_scores)

    # ---------- methods for training ------------
    def _set_lr(self):
        ''' sets learning rates of optimizers according to Flower setting '''
        for n in range(self.nb_nodes): 
            self.opt_nodes[n].param_groups[0]['lr'] = self.lr_node
        self.opt_gen.param_groups[0]['lr'] = self.lr_gen

    def _zero_opt(self):
        ''' resets gradients of all models '''
        for n in range(self.nb_nodes):
            self.opt_nodes[n].zero_grad()      
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
        for i in range(self.nb_nodes):
            self.age[i] += years

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
            for n in range(self.nb_nodes): 
                self.opt_nodes[n].step()      
        else:
            self.opt_gen.step()  

    def _print_losses(self, tot, fit, gen, reg):
        ''' prints losses '''
        print("total loss : ", tot) 
        print("fitting : ", round_loss(fit, 2),
                ', generalisation : ', round_loss(gen, 2),
                ', regularisation : ', round_loss(reg, 2))

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
        # fit_scale = const / self.nb_nodes
        # gen_scale = const / self.nb_nodes / self.size # self.weights is used addition
        # reg_scale = const * self.w0 / self.size

        fit_scale = const 
        gen_scale = const  # self.weights is used addition
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
                    fit_loss, gen_loss = 0, 0
                    for n in range(self.nb_nodes):   # for each node
                        s = torch.ones(1)  # user notation style, constant for now
                        fit_loss += node_local_loss(self.models[n], self.s_nodes[n],  
                                                                    self.data[n][0],
                                                                    self.data[n][1], 
                                                                    self.data[n][2])
                        g = models_dist(self.models[n], 
                                        self.general_model, 
                                        self.pow_gen, 
                                        self.data[n][4] # mask
                                        #None
                                        ) 
                        gen_loss +=  self.weights[n] * g  # generalisation term
                    fit_loss *= fit_scale
                    gen_loss *= gen_scale
                    loss = fit_loss + gen_loss 
                          
                # only last 2 terms of loss updated 
                else:        
                    gen_loss, reg_loss = 0, 0
                    for n in range(self.nb_nodes):   # for each node
                        g = models_dist(self.models[n], 
                                        self.general_model, 
                                        self.pow_gen,
                                        self.data[n][4] # mask
                                        #None
                                        )
                        gen_loss += self.weights[n] * g  # generalisation term    
                    reg_loss = model_norm(self.general_model, self.pow_reg) 
                    gen_loss *= gen_scale
                    reg_loss *= reg_scale
                    loss = gen_loss + reg_loss

                total_out = round_loss(fit_loss + gen_loss + reg_loss)
                if verb >= 2:
                    self._print_losses(total_out, fit_loss, gen_loss, reg_loss)
                # Gradient descent 
                loss.backward() 
                self._do_step(fit_step)   
 
            if verb: print("epoch time :", round(time() - time_ep, 2)) 
            self._update_hist(epoch, fit_loss, gen_loss, reg_loss, verb)
            self._old(1)  # aging all nodes of 1 epoch
             
        # ----------------- end of training -------------------------------  
        print("training time :", round(time() - time_train, 2)) 
        return self.history

    # ------------ to check for problems --------------------------
    def check(self):
        ''' perform some tests on internal parameters adequation '''
        # population check
        b1 =  (self.nb_nodes == len(self.data)  
            == len(self.models) == len(self.opt_nodes) 
            == len(self.weights) == len(self.age))
        # history check
        b2 = True
        for l in self.history:
            b2 = b2 and (len(l) == len(self.history[0]) >= max(self.age))
        if (b1 and b2):
            print("No Problem")
        else:
            print("Coherency problem in Flower object ")


def get_flower(nb_vids, dic, gpu=False, **kwargs):
    ''' get a Flower (ml decentralized structure)
    nb_vids: number of different videos rated by at least one contributor for this criteria
    dic: dictionnary of {vID: idx}
    '''
   # if gpu:
    #    return Flower(test_gpu, gpu=gpu, **kwargs)
    #else:
     #   return Flower(test, gpu=gpu, **kwargs)
    return Flower(nb_vids, dic, gpu=gpu, **kwargs)

