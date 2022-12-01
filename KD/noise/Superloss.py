import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.special import lambertw

class SuperLoss(nn.Module):

    def __init__(self, C=10, lam=1, batch_size=128):
        super(SuperLoss, self).__init__()
        self.tau = math.log(C)
        self.lam = lam  # set to 1 for CIFAR10 and 0.25 for CIFAR100
        self.batch_size = batch_size
                  
    def forward(self, logits, teacher_outputs,targets,ones):
        Temp = 4
        #print(F.cross_entropy(logits, targets, reduction='none').shape,nn.KLDivLoss(reduction='none')(F.log_softmax(logits / Temp, dim=1), F.softmax(teacher_outputs / Temp, dim=1)).shape)
        l_i = ones[:,0]*F.cross_entropy(logits, targets, reduction='none').detach() + \
        Temp*Temp*ones[:,1]*torch.sum(nn.KLDivLoss(reduction='none')(F.log_softmax(logits / Temp, dim=1), \
        F.softmax(teacher_outputs / Temp, dim=1)), dim=1).detach()
                            
        sigma = self.sigma(l_i)
        loss = (ones[:,0]*F.cross_entropy(logits, targets, reduction='none')  + Temp*Temp*ones[:,1]*\
        torch.sum(nn.KLDivLoss(reduction='none')(F.log_softmax(logits / Temp, dim=1), \
        F.softmax(teacher_outputs / Temp, dim=1)), dim=1) - self.tau)*sigma \
        +self.lam*(torch.log(sigma)**2)
        loss = loss.sum()/self.batch_size
        return loss

    def sigma(self, l_i):
        x = torch.ones(l_i.size())*(-2/math.exp(1.))
        x = x.cuda()
        y = 0.5*torch.max(x, (l_i-self.tau)/self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma