import os
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch

import seaborn as sns
import pandas as pd

from torch.distributions import Categorical
import torch.nn.functional as F

targets = np.load('results/No-curr_distil/cifar100/WRN_16_X_16_1_p0/16/targets.npy')
#np.load('results/No-curr_distil/cifar10/WRN_16_X_16_1_p0/16/targets.npy')

tdir = 'results/No-curr_distil/cifar100/'

tmodel = ['WRN_16_X_16_3_p0/16']#, 'WRN_16_X_16_4_p0/16','WRN_16_X_16_3_p0/16','WRN_16_X_16_4_p0/16']

sdir = 'results/MultiLam_distilT/cifar100/'

#smodel = ['WRN_16_X_[16]_[3]_16_1_p1.0/24', 'WRN_16_X_[16]_[4]_16_1_p9.0/24','WRN_16_X_[16]_[6]_16_1_p9.0/24',\
#'WRN_16_X_[16]_[8]_16_1_p9.0/24']

smodel = ['WRN_16_X_[16, 16, 16, 16, 16]_[3, 3, 3, 3, 3]_16_1_p1.0/24']

#smodel = ['WRN_16_X_[16, 16, 16, 16, 16]_[3, 3, 3, 3, 3]_16_1_p1.0/24', \
#'WRN_16_X_[16, 16, 16, 16, 16]_[4, 4, 4, 4, 4]_16_1_p1.0/24','WRN_16_X_[16, 16, 16, 16, 16]_[6, 6, 6, 6, 6]_16_1_p1.0/24',\
# 'WRN_16_X_[16, 16, 16, 16, 16]_[8, 8, 8, 8, 8]_16_1_p1.0/24']


for tm in range(len(tmodel)):

    stou = []

    plt.figure()

    tcheckpoint = torch.load(os.path.join(tdir, tmodel[tm], "model.pt"))

    ftout = tcheckpoint['output'][-1].cpu()
    
    scheckpoint = torch.load(os.path.join(sdir, smodel[tm], "model.pt"))

    sout = scheckpoint['output']
    all_lambda = scheckpoint['lambda']
    #print(all_lambda[10],all_lambda[11])
    epoch = [10]#for i in range(10,len( all_lambda ),10)]

    #nlam = 10
    n=2
    ctype = "pL"#str(n)+"_max"
    

    if ctype == "entropy":
        tcscore = Categorical(probs=F.softmax(ftout/4, dim=1)).entropy().cpu()
        scscore = []
        for i in epoch:
            scscore.append(Categorical(probs=F.softmax(sout[i]/4, dim=1)).entropy().cpu())
    
    elif ctype == "pL":

        tcscore = F.softmax(ftout, dim=1)[torch.arange(ftout.shape[0]),targets].cpu()
        scscore = []
        for i in epoch:
            print(len(sout))
            scscore.append(F.softmax(sout[i], dim=1)[torch.arange(sout[i].shape[0]),targets].cpu())

    elif ctype == str(n)+"_max":

        tcscore = (torch.topk(F.softmax(ftout/4, dim=1).cpu(),10,dim=1).values)[:,n-1]
        scscore = []
        for i in epoch:
            scscore.append((torch.topk(F.softmax(sout[i-1]/4, dim=1).cpu(),10,dim=1).values)[:,n-1])

    os.makedirs(os.path.join(sdir, smodel[tm], "plot",ctype+"_time"), exist_ok=True)

    values = [scscore[0].min().item()]
    #values = [tcscore.min().item()]
    for qu in range(1, 10):
        values.append(torch.quantile(scscore[0], qu / 10).item())
        #values.append(torch.quantile(tcscore, qu / 10).item())

    values.append(scscore[0].max().item())
    #values.append(tcscore.max().item())

    print(values)

    indices = []
    for qu in range(len(values)-1):
        indices.append(torch.logical_and(scscore[0] >= values[qu], scscore[0] <  values[qu+1] ))
        #indices.append(torch.logical_and(tcscore >= values[qu], tcscore <  values[qu+1] ))

    #time = [i+1 for i in range(10,len( all_lambda ),10)][:-1]
    time = [i for i in range(len(all_lambda))] 

    print(all_lambda[0].shape)
    
    df = pd.DataFrame()
    
    for v in range(len(indices)):

        
        curr_lam = []
        for i in time:
            
            #curr_lam.append(all_lambda[i].cpu()[:,1][indices[v]].mean().item())
            curr_lam.append(torch.quantile(all_lambda[i].cpu()[:,1][indices[v]],0.5).item())

        df[str(round(values[v], 2)) + "-" + str(round(values[v+1], 2))] = curr_lam

    df.index = [(i+1)*10 for i in time] #time
    #print(df)
    ax = sns.lineplot(data=df)
    ax.set(xlabel="Epoch", ylabel="Lambdas",)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5),title="Student_"+ctype)#"Teacher_"+ctype

    
    plt.savefig(os.path.join(sdir, smodel[tm], "plot",ctype+"_time",'10.jpg'),bbox_inches='tight')
    #plt.show()
    plt.close()

        

    