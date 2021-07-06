import torch

def predict(input, tens):
    ''' 
    tens: tensor = model
    input: tensor one-hot encoding video

    Returns: - score of the video according to the model
    '''
    return torch.matmul(input.float(), tens)

# losses (used in licchavi.py)
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
    return (0.5 * s**2 - torch.log(s))

def node_local_loss(model, s, a_batch, b_batch, r_batch):
    ''' fitting loss for one node, includes s_loss '''
    ya_batch = predict(a_batch, model)
    yb_batch = predict(b_batch, model)
    loss = 0 
    for ya,yb,r in zip(ya_batch, yb_batch, r_batch):
        loss += fit_loss(s, ya, yb, r)
    #return loss / len(a_batch) + s_loss(s)
    return loss + s_loss(s) # paper version

def models_dist(model1, model2, pow=(1,1), mask=None):  
    ''' distance between 2 models (l1 by default)

    pow : (internal power, external power)
    '''
    q, p = pow
    if mask is None:
        mask = [torch.ones_like(param) for param in [model1]]
    dist = sum(
                (((theta - rho) * coef)**q).abs().sum() for theta, rho, coef 
                                        in zip([model1], [model2], mask)
                )**p
    return dist

def model_norm(model, pow=(2,1)): 
    ''' norm of a model (l2 squared by default)

     pow : (internal power, external power)
     '''
    q, p = pow
    norm = sum((param**q).abs().sum() for param in [model])**p
    return norm

def round_loss(tens, dec=0): 
    ''' from an input scalar tensor or int/float returns rounded int/float '''
    if type(tens)==int or type(tens)==float:
        return round(tens, dec)
    else:
        return round(tens.item(), dec)