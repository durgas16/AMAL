import numpy as np
from pyrsistent import freeze
from torch.autograd import grad
import torch

import torch.nn.functional as F
import torch.nn as nn

def estimate_grads(trainval_loader, model, criterion,teacher_outputs, device,lambdas):
    
    model.train()
    all_grads = []
    all_targets = []

    Temp = 4

    for i, (input, target, indices) in enumerate(trainval_loader):
        
        #print(i)
        input = input.to(device)
        all_targets.append(target)
        target = target.to(device)
        
        # compute output
        output, feat = model(input,last=True)

        loss_SL = criterion(output, target)
        loss = lambdas[indices,0]*loss_SL
        
        loss_KD = nn.KLDivLoss(reduction='none')(F.log_softmax(output / Temp, dim=1),\
             F.softmax(teacher_outputs[indices] / Temp, dim=1))
        loss +=  Temp * Temp *lambdas[indices,1]*torch.sum(loss_KD, dim=1)
            
        loss = torch.mean(loss)
        #output = model.fc(feat)
        #loss = lambdas[indices,0]*criterion(output, target) + lambdas[indices,1]*nn.KLDivLoss(reduction='none')(\
        #F.log_softmax(output / Temp, dim=1), F.softmax(teacher_outputs / Temp, dim=1))
        
        est_grad = grad(loss, feat)
        all_grads.append(est_grad[0].detach().cpu().numpy())
        #print(est_grad[0].shape)
        
        '''l0_grads = (grad(loss, outputs)[0]).detach().clone().cuda(1)
        l0_expand = torch.repeat_interleave(l0_grads, feat.shape[1], dim=1)
        l1_grads = l0_expand * feat.repeat(1, output.shape[1]).cuda(1)
        all_grads.append(est_grad[0].detach().cpu().numpy())'''

    all_grads = np.vstack(all_grads)
    all_targets = np.hstack(all_targets)
   
    return all_grads, all_targets