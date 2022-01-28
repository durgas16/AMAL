import os
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch

import seaborn as sns
import pandas as pd

from torch.distributions import Categorical
import torch.nn.functional as F

#targets = np.load('../results/No-curr_distil/airplane/WRN_16_X_16_8_p0/16/targets.npy')
#final_model = torch.load("../results/No-curr_distil/airplane/WRN_16_X_16_8_p0/16/model.pt")

color_list = [(1,0.6,0),(0,0.8,0)]

targets = np.load('../results/No-curr_distil/synthetic/NN_2L_80_p0/class/None/24/targets.npy')
final_model = torch.load("../results/No-curr_distil/synthetic/NN_2L_80_p0/class/None/24/model.pt")

#reout = final_model['output'][40].cpu()
#eout = (F.softmax(reout, dim=1))[torch.arange(reout.shape[0]),targets].cpu()
#eout = Categorical(probs=F.softmax(reout, dim=1)).entropy().cpu()
rfout = final_model['output'][-1].cpu()
fout = (F.softmax(rfout, dim=1))[torch.arange(rfout.shape[0]),targets].cpu()
#fout = Categorical(probs=F.softmax(rfout, dim=1)).entropy().cpu()

#learnt_model = torch.load('../results/MultiLam_distilT/airplane/WRN_16_X_[16]_[8]_16_8_p1.0/16_o/model.pt')
seeds = [67,8,13,16,18,20,24,36,42,26,32,55,84,9,75]

learnt_model = ['../results/MultiLam_distilT/synthetic/NN_2L_50_p1.0/class/None/'+str(i)+'/model.pt'\
for i in seeds]

for v in range(len(learnt_model)):
    learnt = torch.load(learnt_model[v])
    #print(len(learnt['metrics']['tst_acc']))
    #print(seeds[v])
    if v == 0:
        test_acc = torch.tensor(learnt['lambda'][-1][:,1].cpu()).view(-1,1)
    else:
        test_acc = torch.cat((test_acc,torch.tensor(learnt['lambda'][-1][:,1].cpu()).view(-1,1)),dim =1)

#print(test_acc[0])

noise_lb = learnt['noise_lb']
noise_bool = torch.zeros(len(learnt['lambda'][0])).bool()
noise_bool[noise_lb] = True
#print(flambda[noise_bool].shape)
#print(flambda.max(), flambda.min(), torch.median(flambda), torch.quantile(flambda, 0.75),torch.quantile(flambda, 0.25))

'''lam_values_noise = [flambda[noise_bool].min().item()]
lam_values = [flambda[~noise_bool].min().item()]
for qu in range(1, 10):
    lam_values.append(torch.quantile(flambda[~noise_bool], qu / 10).item())
    lam_values_noise.append(torch.quantile(flambda[noise_bool], qu / 10).item())
lam_values.append(flambda[~noise_bool].max().item())
lam_values_noise.append(flambda[~noise_bool].max().item())'''

lam_values_noise = [fout[noise_bool].min().item()]
lam_values = [fout[~noise_bool].min().item()]

divide = 100
for qu in range(1, divide):
    lam_values.append(torch.quantile(fout[~noise_bool], qu / divide).item())
    lam_values_noise.append(torch.quantile(fout[noise_bool], qu / divide).item())
lam_values.append(fout[~noise_bool].max().item())
lam_values_noise.append(fout[~noise_bool].max().item())

plt.figure()
kwargs = dict(alpha=0.7, bins=10,linewidth=1,ec='k')
plt.grid(axis='y',linestyle='--', linewidth=1)
print(len(fout[~noise_bool]))
plt.hist(fout[~noise_bool].numpy(), **kwargs,color=color_list[1],label="Clean")
plt.hist(fout[noise_bool].numpy(), **kwargs,color=color_list[0],label="Noise")
plt.legend()
plt.title("Convergered teacher probability of true labels \n for clean and noisy lables")
plt.xlabel('Convergered teacher probability of true labels')
plt.ylabel('No, of training points')
plt.savefig('./plots/teach_prob.jpg')

#print(lam_values)
#print(lam_values_noise)

lam_indices =[]
lam_indices_noise =[]
'''for qu in range(len(lam_values)-1):
    lam_indices.append(torch.logical_and(flambda[~noise_bool] >= lam_values[qu], flambda[~noise_bool]< lam_values[qu+1] ))
    lam_indices_noise.append(torch.logical_and(flambda[noise_bool] >= lam_values_noise[qu], flambda[noise_bool]< lam_values_noise[qu+1] ))
'''

for qu in range(len(lam_values)-1):
    lam_indices.append(torch.logical_and(fout[~noise_bool] >= lam_values[qu], fout[~noise_bool]< lam_values[qu+1] ))
    lam_indices_noise.append(torch.logical_and(fout[noise_bool] >= lam_values_noise[qu], fout[noise_bool]< lam_values_noise[qu+1] ))

bfout = []
bnout = []
mean_flambda = torch.zeros((divide,len(seeds)))
mean_flambda_noise = torch.zeros((divide,len(seeds)))
'''std_flambda_upper = []
std_flambda_lower = []
std_flambda_noise_lower = []
std_flambda_noise_upper = []'''

#print(flambda.shape,stdlist.shape)
for v in range(len(lam_indices)):
    bfout.append((fout[~noise_bool][lam_indices[v]]).mean().item())
    bnout.append((fout[noise_bool][lam_indices_noise[v]]).mean().item())
    
    for s in range(len(seeds)):
        mean_flambda[v][s] = (test_acc[:,s][~noise_bool][lam_indices[v]]).mean().item()
        mean_flambda_noise[v][s] = (test_acc[:,s][noise_bool][lam_indices_noise[v]]).mean().item()

    '''std_flambda_upper.append(mean_flambda[-1]+(stdlist[~noise_bool][lam_indices[v]]).mean().item())
    std_flambda_lower.append(mean_flambda[-1]-(stdlist[~noise_bool][lam_indices[v]]).mean().item())
    
    std_flambda_noise_upper.append(mean_flambda_noise[-1]+(stdlist[noise_bool][lam_indices_noise[v]]).mean().item())
    std_flambda_noise_lower.append(mean_flambda_noise[-1]-(stdlist[noise_bool][lam_indices_noise[v]]).mean().item())'''

#print(torch.std_mean((flambda[noise_bool])[flambda[noise_bool] > 0.5], unbiased=False),\
#torch.std_mean((flambda[~noise_bool])[flambda[~noise_bool] > 0.5], unbiased=False))

#print(std_flambda_upper)
#print(std_flambda_lower)
#print(len(bfout),mean_flambda)
stdlist,avglist = torch.std_mean(mean_flambda,dim=1,unbiased=True)
stdlist_noise,avglist_noise = torch.std_mean(mean_flambda_noise,dim=1,unbiased=True)
print(stdlist.shape)

plt.figure()
plt.grid(axis='y',linestyle='--', linewidth=1)
plt.scatter( bnout,avglist_noise ,label="Noise",color=color_list[0])
plt.fill_between( bnout, avglist_noise + stdlist_noise, avglist_noise - stdlist_noise, alpha=0.2, color=color_list[0])
plt.scatter(bfout,avglist , label="Clean", color=color_list[1])
plt.fill_between( bfout, avglist + stdlist, avglist - stdlist, alpha=0.2, color=color_list[1])
plt.legend()
plt.title("Lambda values for clean and noisy lables")
plt.xlabel('Convergered teacher probability of true labels')
plt.ylabel('Lambdas')
plt.savefig('./plots/clean_vs_noise.jpg')


