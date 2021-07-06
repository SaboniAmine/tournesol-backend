from .plots import loss_var 

def licch_stats(licch):
    ''' gives some statistics about licchavi object '''
    licch.check() # some tests
    h = licch.history
    print("nb_nodes", licch.nb_nodes)
    licch.stat_s()
    loss_var([h], path="ml/")
