import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

import faulthandler


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class LearnMultiLambdaMeta(object):
    """
    Implementation of Data Selection Strategy class which serves as base class for other
    dataselectionstrategies for general learning frameworks.
    Parameters
        ----------
        trainloader: class
            Loading the training data using pytorch dataloader
        model: class
            Model architecture used for training
        num_classes: int
            Number of target classes in the dataset
    """

    def __init__(self, trainloader, valloader, model, num_classes, N_trn, loss, device, fit, \
                 teacher_model, criterion_red, temp):
        """
        Constructer method
        """
        self.trainloader = trainloader  # assume its a sequential loader.
        self.valloader = valloader
        self.model = model
        self.N_trn = N_trn

        self.num_classes = num_classes
        self.device = device

        self.fit = fit
        # self.batch = batch
        # self.dist_batch = dist

        self.teacher_model = teacher_model
        self.criterion = loss
        self.criterion_red = criterion_red
        self.smoothLoss = LabelSmoothingLoss(num_classes, smoothing=0.1)
        self.squareLoss = loss = nn.MSELoss()
        self.temp = temp
        print(N_trn)

    def update_model(self, model_params):
        """
        Update the models parameters

        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        """
        self.model.load_state_dict(model_params)

    
    def get_lambdas(self, eta,lam):

        offset = 0
        batch_wise_indices = list(self.trainloader.batch_sampler)
        #eta =0.1

        #self.smoothLoss.smoothing = 0.4 
        lambdas = lam.cuda(1)#, device=self.device)
        #soft_lam = F.softmax(lambdas, dim=1)

        with torch.no_grad():

            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    self.init_out = out
                    self.init_l1 = l1
                    self.y_val = targets  # .view(-1, 1)
                    tea_out_val = self.teacher_model[-1](inputs)
                else:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    self.init_out = torch.cat((self.init_out, out), dim=0)
                    self.init_l1 = torch.cat((self.init_l1, l1), dim=0)
                    self.y_val = torch.cat((self.y_val, targets), dim=0)
                    tea_out_val = torch.cat((tea_out_val, self.teacher_model[-1](inputs)), dim=0)

            # val_loss_SL = self.criterion_red(self.init_out,self.y_val)

            #val_loss_SL = torch.sum(self.criterion(self.init_out, self.y_val))

            # print(val_loss_SL)

            '''val_loss_KD = nn.KLDivLoss(reduction='none')(F.log_softmax(self.init_out / self.temp, dim=1),\
                F.softmax(tea_out_val / self.temp, dim=1))
            val_loss_KD = self.temp*self.temp*torch.sum (val_loss_KD)#torch.mean(torch.sum (val_loss_KD,dim=1))'''

            # val_loss_KD = self.temp*self.temp*nn.KLDivLoss(reduction='batchmean')(F.log_softmax(self.init_out / self.temp, dim=1),\
            #    F.softmax(tea_out_val / self.temp, dim=1))

            self.init_out = self.init_out.cuda(1)
            self.init_l1 = self.init_l1.cuda(1)
            self.y_val = self.y_val.cuda(1)
            tea_out_val = tea_out_val.cuda(1)

        KD_grads = [0 for _ in range(len(self.teacher_model))]
        c_temp = self.temp
        for batch_idx, (inputs, target,indices) in enumerate(self.trainloader):

            #batch_wise_indices = list(self.trainloader.batch_sampler)

            inputs, target = inputs.to(self.device), target.to(self.device, non_blocking=True)

            outputs, l1 = self.model(inputs, last=True, freeze=True)
            
            loss_SL = self.criterion_red(outputs, target)  # self.criterion(outputs, target).sum()

            l0_grads = (torch.autograd.grad(loss_SL, outputs)[0]).detach().clone().cuda(1)
            l0_expand = torch.repeat_interleave(l0_grads, l1.shape[1], dim=1)
            l1_grads = l0_expand * l1.repeat(1, self.num_classes).cuda(1)

            if batch_idx % self.fit == 0:
                with torch.no_grad():
                    train_out = outputs.cuda(1)
                    train_l1 = l1.cuda(1)
                    train_target = target.cuda(1)
                    tea_out = self.teacher_model[-1](inputs).cuda(1)
                SL_grads = torch.cat((l0_grads, l1_grads), dim=1)
                batch_ind = list(indices) #batch_wise_indices[batch_idx]
            else:
                with torch.no_grad():
                    train_out = torch.cat((train_out,outputs.cuda(1)), dim=0)
                    train_l1 = torch.cat((train_l1,l1.cuda(1)), dim=0)
                    train_target = torch.cat((train_target,target.cuda(1)), dim=0)
                    tea_out = torch.cat((tea_out, self.teacher_model[-1](inputs).cuda(1)), dim=0)
                SL_grads = torch.cat((SL_grads, torch.cat((l0_grads, l1_grads), dim=1)), dim=0)
                batch_ind.extend(list(indices))#batch_wise_indices[batch_idx])

            for m in range(len(self.teacher_model)):
                with torch.no_grad():
                    teacher_outputs = self.teacher_model[m](inputs)
                loss_KD = self.temp * self.temp * nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(outputs / self.temp, dim=1), \
                    F.softmax(teacher_outputs / self.temp, dim=1))

                #loss_KD = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs, dim=1), \
                #    F.softmax(teacher_outputs , dim=1))

                #loss_KD = torch.mean(torch.sum(-F.softmax(teacher_outputs, dim=1) * \
                #F.log_softmax(outputs, dim=1),dim=1))

                l0_grads = (torch.autograd.grad(loss_KD, outputs)[0]).detach().clone().cuda(1)
                l0_expand = torch.repeat_interleave(l0_grads, l1.shape[1], dim=1)
                l1_grads = l0_expand * l1.repeat(1, self.num_classes).cuda(1)

                if batch_idx % self.fit == 0:
                    KD_grads[m] = torch.cat((l0_grads, l1_grads), dim=1)
                else:
                    KD_grads[m] = torch.cat((KD_grads[m], torch.cat((l0_grads, l1_grads), dim=1)), dim=0)

            if (batch_idx + 1) % self.fit == 0 or batch_idx + 1 == len(self.trainloader):

                for r in range(10):
                    #print("Before",lambdas[batch_ind[0]].item(),lambdas[batch_ind[-1]].item())
                    
                    #comb_grad = soft_lam[batch_ind,0][:,None]*SL_grads 
                    comb_grad = lambdas[batch_ind,0][:,None]*SL_grads 
                    
                    for m in range(len(self.teacher_model)):
                        #comb_grad += soft_lam[batch_ind,m+1][:,None]*KD_grads[m]
                        comb_grad += lambdas[batch_ind,m+1][:,None]*KD_grads[m]

                    comb_grad = comb_grad.sum(0)

                    out_vec_val = self.init_out - (eta * comb_grad[:self.num_classes].view(1, -1).\
                        expand(self.init_out.shape[0], -1))

                    out_vec_val = out_vec_val - (eta * torch.matmul(self.init_l1, comb_grad[self.num_classes:].\
                        view(self.num_classes, -1).transpose(0, 1)))

                    out_vec_val.requires_grad = True
                    loss_SL_val = self.smoothLoss(out_vec_val, self.y_val)  # self.criterion(outputs, target).sum()

                    loss_KD_val = c_temp * c_temp *nn.KLDivLoss(reduction='batchmean')(F.log_softmax(\
                    out_vec_val/c_temp , dim=1), F.softmax(tea_out_val/c_temp, dim=1))

                    #l0_grads = (torch.autograd.grad(0.25*loss_KD_val+0.75*loss_SL_val, out_vec_val)[0]).detach().\
                    #clone().cuda(1)
                    l0_grads = (torch.autograd.grad(loss_SL_val, out_vec_val)[0]).detach().clone().cuda(1)
                    #print(round(loss_KD_val.item(),4), end=",")

                    #print(round(loss_SL_val.item(),4), end=",")
                    l0_expand = torch.repeat_interleave(l0_grads, self.init_l1.shape[1], dim=1)
                    l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes).cuda(1)
                    up_grads_val = torch.cat((l0_grads, l1_grads), dim=1).sum(0)

                    out_vec = train_out - (eta * comb_grad[:self.num_classes].view(1, -1).expand(train_out.shape[0], -1))

                    out_vec = out_vec - (eta * torch.matmul(train_l1, comb_grad[self.num_classes:].\
                        view(self.num_classes, -1).transpose(0, 1)))

                    out_vec.requires_grad = True

                    loss_SL = self.criterion_red(out_vec, train_target) #self.smoothLoss(out_vec, train_target)   

                    loss_KD = c_temp * c_temp *nn.KLDivLoss(reduction='batchmean')(F.log_softmax(\
                    out_vec/c_temp , dim=1), F.softmax(tea_out/c_temp, dim=1))

                    if (batch_idx + 1) % (self.fit*40) ==0:
                        print(round(loss_SL_val.item(),4),"+",round(loss_KD_val.item(),4),"+",\
                        round(loss_SL.item(),4),"+",round(loss_KD.item(),4), end=",")
                        #print(round(loss_SL_val.item(),4),"+",round(loss_SL.item(),4), end=",")

                    #l0_grads = (torch.autograd.grad(0.25*loss_KD+0.75*loss_SL, out_vec)[0]).detach().clone().cuda(1)
                    l0_grads = (torch.autograd.grad(loss_SL, out_vec)[0]).detach().clone().cuda(1)
                    l0_expand = torch.repeat_interleave(l0_grads, train_l1.shape[1], dim=1)
                    l1_grads = l0_expand * train_l1.repeat(1, self.num_classes).cuda(1)
                    up_grads = torch.cat((l0_grads, l1_grads), dim=1).sum(0)

                    
                    #combined = (0.9*up_grads_val+0.1*up_grads).T
                    #if eta > 0.02:
                    #    combined = (0.1*up_grads_val+0.9*up_grads).T
                    #else:
                    combined = (0.9*up_grads_val+0.1*up_grads).T

                    """tea_val = torch.max(F.log_softmax(tea_out_val/c_temp, dim=1),dim=1).values
                    stu_val = F.log_softmax(out_vec_val/c_temp, dim=1)[torch.arange(out_vec_val.shape[0]),self.y_val]
                    loss_val = self.squareLoss(tea_val,stu_val)

                    if (batch_idx + 1) % (self.fit*40) ==0:
                        print(round(loss_val.item(),4), end=",")

                    l0_grads = (torch.autograd.grad(loss_val, out_vec_val)[0]).detach().clone().cuda(1)
                    l0_expand = torch.repeat_interleave(l0_grads, self.init_l1.shape[1], dim=1)
                    l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes).cuda(1)
                    combined = torch.cat((l0_grads, l1_grads), dim=1).sum(0)"""

                    '''grad = ((1-soft_lam[batch_ind,0])*soft_lam[batch_ind,0])[:,None]*SL_grads
                    #grad = SL_grads 
                    for m_1 in range(len(self.teacher_model)):
                        grad -= (soft_lam[batch_ind,0]*soft_lam[batch_ind,m_1+1])[:,None]*KD_grads[m_1]
                        #grad -= KD_grads[m_1]
                    alpha_grads = torch.matmul(grad,combined)
                    lambdas[batch_ind,0] = lambdas[batch_ind,0] +  6000*eta*alpha_grads #9*eta*'''
                    
                    for m in range(len(self.teacher_model)):
                        #grad = (-soft_lam[batch_ind,0]*soft_lam[batch_ind,m+1])[:,None]*SL_grads 
                        grad = KD_grads[m] -SL_grads 
                        '''for m_1 in range(len(self.teacher_model)):
                            if m_1 == m:
                                #grad += ((1-soft_lam[batch_ind,m_1+1])*soft_lam[batch_ind,m_1+1])[:,None]*KD_grads[m_1]
                                grad += KD_grads[m_1]
                            else:
                                #grad -= (soft_lam[batch_ind,m+1]*soft_lam[batch_ind,m_1+1])[:,None]*KD_grads[m_1]
                                grad -= KD_grads[m_1]'''
                        alpha_grads = torch.matmul(grad,combined)
                        lambdas[batch_ind,m+1] = lambdas[batch_ind,m+1] +  1500*eta*alpha_grads #9*eta*
                    
                    #print("After",lambdas[batch_ind[0]].item(),lambdas[batch_ind[-1]].item())
                    lambdas.clamp_(min=1e-7,max=1-1e-7)
                    #soft_lam = F.softmax(lambdas, dim=1)
                    lambdas[batch_ind,0] = 1- torch.max(lambdas[batch_ind,1:],dim=1).values
                if (batch_idx + 1) % (self.fit*40) ==0:
                    print()#"End for loop")

        #lambdas.clamp_(min=0.01,max=0.99)
        return lambdas.cuda(0)



