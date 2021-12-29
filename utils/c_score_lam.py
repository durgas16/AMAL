import os
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch

import seaborn as sns
import pandas as pd

from torch.distributions import Categorical
import torch.nn.functional as F

targets = np.load('results/No-curr_distil/cifar10/WRN_16_X_16_1_p0/16/targets.npy')

tdir = 'results/No-curr_distil/cifar10/'

tmodel = ['WRN_16_X_16_3_p0/16', 'WRN_16_X_16_4_p0/16','WRN_16_X_16_3_p0/16','WRN_16_X_16_4_p0/16']

sdir = 'results/MultiLam_distilT/cifar10/'

smodel = ['WRN_16_X_[16]_[3]_16_1_p1.0/24', 'WRN_16_X_[16]_[4]_16_1_p1.0/24','WRN_16_X_[16]_[6]_16_1_p1.0/24',\
'WRN_16_X_[16]_[8]_16_1_p1.0/24']


for tm in range(len(tmodel)):

    stou = []

    plt.figure()

    tcheckpoint = torch.load(os.path.join(tdir, tmodel[tm], "model.pt"))

    ftout = tcheckpoint['output'][-1].cpu()
    
    scheckpoint = torch.load(os.path.join(sdir, smodel[tm], "model.pt"))

    sout = scheckpoint['output']
    all_lambda = scheckpoint['lambda']
    #print(all_lambda[10],all_lambda[11])
    epoch = [50,80,100,120,150,len(all_lambda)-1]

    nlam = 10
    n=2
    ctype = str(n)+"_max"
    

    if ctype == "entropy":
        tcscore = Categorical(probs=F.softmax(ftout/4, dim=1)).entropy().cpu()
        scscore = []
        for i in epoch:
            scscore.append(Categorical(probs=F.softmax(sout[i]/4, dim=1)).entropy().cpu())
    
    elif ctype == "pL":

        tcscore = F.softmax(ftout/4, dim=1)[torch.arange(ftout.shape[0]),targets].cpu()
        scscore = []
        for i in epoch:
            scscore.append(F.softmax(sout[i]/4, dim=1)[torch.arange(sout[i].shape[0]),targets].cpu())

    elif ctype == str(n)+"_max":

        tcscore = (torch.topk(F.softmax(ftout/4, dim=1).cpu(),10,dim=1).values)[:,n-1]
        scscore = []
        for i in epoch:
            scscore.append((torch.topk(F.softmax(sout[i-1]/4, dim=1).cpu(),10,dim=1).values)[:,n-1])

    os.makedirs(os.path.join(sdir, smodel[tm], "plot",ctype), exist_ok=True)

    for e in range(len(epoch)):

        curr_lam = all_lambda[epoch[e]].cpu()[:,1] #F.softmax(all_lambda[epoch[e]].cpu() , dim=1)[:,1]
        #g_lam = torch.ones_like(curr_lam)*curr_lam.min().item()
        g_lam = torch.zeros_like(curr_lam)
        values = [curr_lam.min().item()]
        for qu in range(1, 10):
            #g_lam[curr_lam > torch.quantile(curr_lam, qu / 10)] = torch.quantile(curr_lam, qu / 10)
            #values.append(torch.quantile(curr_lam, qu / 10).item())
            g_lam[curr_lam > qu/10] = qu/10
            values.append(qu/10)

        #values.append(curr_lam.max().item())
        values.append(1.0)
        
        #index_1 = torch.logical_and(curr_lam >= values[nlam-1], curr_lam <=  values[nlam] )
        #index_2 = torch.logical_and(curr_lam >= values[0], curr_lam < values[1])

        #curr_lam = all_lambda[21].cpu()[:,1]
        index_1 = torch.logical_and(curr_lam >= 0.7, curr_lam <=  0.99)
        index_2 = torch.logical_and(curr_lam >= 0.01, curr_lam < 0.2)
    
        index = torch.cat(((index_1).nonzero().view(-1), (index_2).nonzero().view(-1)))

        print(values)

        df = pd.DataFrame()
        df["Teacher_"+ctype] = tcscore[index]
        df["Student_"+ctype] = scscore[e][index]
        #df["g_lam"] = [str(round(qu, 2)) + "-" + str(round(values[values.index(qu)+1], 2)) for qu in g_lam[index].tolist()] #
        df["g_lam"] = [str(round(qu, 2)) + "-" + str(round(qu+0.1, 2)) for qu in g_lam[index].tolist()]
        df.sort_values(by=['g_lam'])
    
        sns.scatterplot(x="Teacher_"+ctype, y="Student_"+ctype, data=df , hue="g_lam", \
                        palette=sns.color_palette("Paired", len(df["g_lam"].unique()))).set(
            title="Lambdas @ " + str(epoch[e]) + " epoch")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5),title="Lambdas")

        """#curr_lam = tcscore
        curr_lam = scscore[e]
        g_lam = torch.ones_like(curr_lam)*curr_lam.min().item()
        values = [curr_lam.min().item()]
        for qu in range(1, 10):
            g_lam[curr_lam > torch.quantile(curr_lam, qu / 10)] = torch.quantile(curr_lam, qu / 10)
            values.append(torch.quantile(curr_lam, qu / 10).item())

        values.append(curr_lam.max().item())
        print(g_lam)

        index_1 = torch.logical_and(curr_lam >= values[-2], curr_lam <=  values[-1] -0.01 )
        index_2 = torch.logical_and(curr_lam >= curr_lam.min().item()-0.01, curr_lam < values[1])
    
        index =  torch.cat(((index_1).nonzero().view(-1), (index_2).nonzero().view(-1)))
        
        print(torch.median(all_lambda[epoch[e]].cpu()[:,1][index_1]),torch.median(all_lambda[epoch[e]].cpu()[:,1][index_2]))
        df = pd.DataFrame()
        df["Lambdas"] = all_lambda[epoch[e]].cpu()[:,1][index] #F.softmax(all_lambda[epoch[e]].cpu() , dim=1)[:,1]
        #df["Student_"+ctype] = scscore[e][index]
        df["Teacher_"+ctype] = tcscore[index]
        df["g_lam"] = [str(round(qu, 2)) + "-" + str(round(values[values.index(qu)+1], 2)) for qu in g_lam[index].tolist()]
        df.sort_values(by=['g_lam'])

        '''sns.scatterplot(x="Student_"+ctype,  y ="Lambdas", data=df, hue="g_lam", \
                        palette=sns.color_palette("Paired", len(df["g_lam"].unique()))).set(
            title="@ " + str(epoch[e]) + " epoch")'''

        sns.scatterplot(x="Teacher_"+ctype,  y ="Lambdas", data=df, hue="g_lam", \
                        palette=sns.color_palette("Paired", len(df["g_lam"].unique()))).set(
            title="@ " + str(epoch[e]) + " epoch")
        
        #plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5),title="Teacher_"+ctype)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5),title="Student_"+ctype)"""
        plt.savefig(os.path.join(sdir, smodel[tm], "plot",ctype,str(epoch[e])+"_"+str(nlam)+'.jpg'),bbox_inches='tight')
        #plt.show()
        plt.close()

        

    