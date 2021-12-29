import os
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch

directory = 'results/RewLam_distilT/cifar100/'

model = ['ResNet8_T_p1.0', 'ResNet18_T_p1.0']

for i in range(len(model)):

    plt.figure()

    checkpoint = torch.load(os.path.join(directory, model[i], "modelT.pt"))

    # print(len(checkpoint['lambda']),len(checkpoint['lambda'][0]))
    # print(checkpoint['lambda'])

    _lambda = torch.zeros((len(checkpoint['lambda']), len(checkpoint['lambda'][0])))

    for j in range(len(checkpoint['lambda'])):
        _lambda[j] = checkpoint['lambda'][j].cpu()

    print(_lambda.shape)

    Fmean = torch.mean(_lambda[:2], dim=0).numpy()
    Lmean = torch.mean(_lambda[-2:], dim=0).numpy()

    print(Lmean.shape, len(model[i]))

    plt.scatter(Fmean, Lmean)

    plt.legend()
    plt.title(model[i][:-5])
    plt.xlabel('Mean of first two lambda values')
    plt.ylabel('Mean of last two lambda values')
    plt.savefig(directory + 'plots/' + model[i][:-5] + '.jpg')

