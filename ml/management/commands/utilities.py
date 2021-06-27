import torch



def get_node_vids(batch):
    ''' creates a tensor containing IDs of videos rated by one user without repetition '''
    ids = torch.unique(batch[:,:2])
    batch_ids = torch.unsqueeze(ids,1)
    return batch_ids

def get_all_vids(l_vids):
    ''' l_vids : list of batches of videos IDs (1 batch/node) '''
    all_ids = torch.vstack(l_vids)
    ids = torch.unique(all_ids)
    batch_ids = torch.unsqueeze(ids,1)
    return batch_ids


# metrics on models
def extract_grad(model):
    '''return list of gradients of a model'''
    l_grad =  [p.grad for p in model.parameters()]
    return l_grad

def sp(l_grad1, l_grad2):
    '''scalar product of 2 lists of gradients'''
    s = 0
    for g1, g2 in zip(l_grad1, l_grad2):
        s += (g1 * g2).sum()
    return round_loss(s, 4)

def nb_params(model):
    '''return number of parameters of a model'''
    return sum(p.numel() for p in model.parameters())

#loss and scoring functions 
def models_dist(model_loc, model_glob, pow=(1,1)):  
    ''' l1 distance between global and local parameter
        will be mutliplied by w_n 
        pow : (internal power, external power)
    '''
    q, p = pow
    dist = sum(((theta - rho)**q).abs().sum() for theta, rho in 
                  zip(model_loc.parameters(), model_glob.parameters()))**p
    return dist

def model_norm(model_glob, pow=(2,1)): 
    ''' l2 squared regularisation of global parameter
     will be multiplied by w_0 
     pow : (internal power, external power)
     '''
    q, p = pow
    norm = sum((param**q).abs().sum() for param in model_glob.parameters())**p
    return norm

def round_loss(tens, dec=0): 
    '''from an input scalar tensor returns rounded integer'''
    if type(tens)==int or type(tens)==float:
        return round(tens, dec)
    else:
        return round(tens.item(), dec)

def tens_count(tens, val):
    ''' counts nb of -val in tensor -tens '''
    return len(tens) - round_loss(torch.count_nonzero(tens-val))

def score(model, datafull):
    ''' returns accuracy provided models, images and GTs '''
    out = model(datafull[0])
    predictions = torch.max(out, 1)[1]
    c=0
    for a, b in zip(predictions, datafull[1]):
        c += int(a==b)
    return c/len(datafull[0])

# losses

def fbbt(t,r):
    ''' fbbt loss function '''
    return torch.log(abs(torch.sinh(t)/t)) + r * t + torch.log(torch.tensor(2))

def hfbbt(t,r):
    ''' approximated fbbt loss function '''
    if abs(t) <= 0.01:
        return t**2 / 6 + r *t + torch.log(torch.tensor(2))
    elif abs(t) < 10:
        return torch.log(2 * torch.sinh(t) / t) + r * t
    else:
        return abs(t) - torch.log(abs(t)) + r * t

def fit_loss(s, ya, yb, r):  
    ''' loss for one comparison '''
    loss = hfbbt(s * (ya - yb), r)   
    return loss

def s_loss(s):
    ''' second term of local loss (for one node) '''
    return (2 * s - torch.log(s))

def node_local_loss(model, s, a_batch, b_batch, r_batch):
    ''' fitting loss for one node, includes s_loss '''
    ya_batch = model(a_batch.float())
    yb_batch = model(b_batch.float())
    loss = 0 
    for ya,yb,r in zip(ya_batch, yb_batch, r_batch):
        loss += fit_loss(s, ya, yb, r)
    return loss / len(a_batch) + s_loss(s)