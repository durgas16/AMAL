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
        self.model.eval()

    
    def get_lambdas(self, eta,epoch,lam):

        self.model.eval()
        for m in range(len(self.teacher_model)):
            self.teacher_model[m].eval()
        with torch.no_grad():
            val_loss = 0
            val_total = 0
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                # print(batch_idx)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion_red(outputs, targets)
                val_loss += targets.size(0)*loss.item()
                val_total += targets.size(0)
                print(val_loss/val_total,end=",")
        print()

        smoothing = 0.2
        
        KD_grads = [0 for _ in range(len(self.teacher_model))]
        '''if epoch > 100:
            c_temp = 1 #self.temp
        else:
            c_temp = self.temp'''
        c_temp = 1#self.temp
        max_value = 2
        dataloader_iterator = iter(self.valloader)
        for batch_idx, (inputs, target,indices) in enumerate(self.trainloader):

            #batch_wise_indices = list(self.trainloader.batch_sampler)
            self.model.zero_grad()
            inputs, target = inputs.to(self.device), target.to(self.device, non_blocking=True)

            outputs, l1 = self.model(inputs, last=True, freeze=True)
            #print(l1.shape)
            loss_SL = self.criterion_red(outputs, target)  # self.criterion(outputs, target).sum()

            l0_grads = (torch.autograd.grad(loss_SL, outputs)[0]).detach().clone().to(self.device)
            l0_expand = torch.repeat_interleave(l0_grads, l1.shape[1], dim=1)
            l1_grads = l0_expand * l1.detach().repeat(1, self.num_classes).to(self.device)

            if batch_idx % self.fit == 0:
                with torch.no_grad():
                    train_out = outputs.detach().to(self.device)
                    train_l1 = l1.detach().to(self.device)
                    train_target = target.to(self.device)
                    tea_out = self.teacher_model[-1](inputs).to(self.device)
                SL_grads = torch.cat((l0_grads, l1_grads), dim=1)
                batch_ind = list(indices) #batch_wise_indices[batch_idx]
            else:
                with torch.no_grad():
                    train_out = torch.cat((train_out,outputs.detach().to(self.device)), dim=0)
                    train_l1 = torch.cat((train_l1,l1.detach().to(self.device)), dim=0)
                    train_target = torch.cat((train_target,target.to(self.device)), dim=0)
                    tea_out = torch.cat((tea_out, self.teacher_model[-1](inputs).to(self.device)), dim=0)
                SL_grads = torch.cat((SL_grads, torch.cat((l0_grads, l1_grads), dim=1)), dim=0)
                batch_ind.extend(list(indices))#batch_wise_indices[batch_idx])
            #print(SL_grads.shape)

            for m in range(len(self.teacher_model)):
                with torch.no_grad():
                    teacher_outputs = self.teacher_model[m](inputs)
                loss_KD = self.temp * self.temp * nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(outputs / self.temp, dim=1), \
                    F.softmax(teacher_outputs / self.temp, dim=1))

                l0_grads = (torch.autograd.grad(loss_KD, outputs)[0]).detach().clone().to(self.device)
                l0_expand = torch.repeat_interleave(l0_grads, l1.shape[1], dim=1)
                l1_grads = l0_expand * l1.detach().repeat(1, self.num_classes).to(self.device)

                if batch_idx % self.fit == 0:
                    KD_grads[m] = torch.cat((l0_grads, l1_grads), dim=1)
                else:
                    KD_grads[m] = torch.cat((KD_grads[m], torch.cat((l0_grads, l1_grads), dim=1)), dim=0)

                #print(batch_idx , self.fit)
                #print(KD_grads[0].shape)
            if (batch_idx + 1) % self.fit == 0 or batch_idx + 1 == len(self.trainloader):

                with torch.no_grad():

                    try:
                        X, Y = next(dataloader_iterator)
                    except:
                        dataloader_iterator = iter(self.valloader)
                        X, Y = next(dataloader_iterator)

                    out, l1 = self.model(X.to(self.device), last=True, freeze=True)
                    self.init_out = out.detach().to(self.device)
                    self.init_l1 = l1.detach().to(self.device)
                    '''for t in range(len(self.teacher_model)):
                        temp = self.teacher_model[t](X.to(self.device)).to(self.device)
                        if t == 0:
                            t_max = self.criterion_red(temp, Y.to(self.device))
                            tea_out_val = temp
                        else:
                            if t_max > self.criterion_red(temp, Y.to(self.device)):
                                t_max = self.criterion_red(temp, Y.to(self.device))
                                tea_out_val = temp'''

                
                for r in range(5):

                    lambdas = lam.clone().to(self.device)#, device=self.device)
                    lambdas.requires_grad = True
                    #print("Before",lambdas[batch_ind[0]].item(),lambdas[batch_ind[-1]].item())
                    
                    #comb_grad = soft_lam[batch_ind,0][:,None]*SL_grads 
                    lambdas1 = lambdas[batch_ind,0][:,None]
                    comb_grad_all = lambdas1*SL_grads 
                    
                    lambdas_2 =[]
                    for m in range(len(self.teacher_model)):
                        lambdas_2.append(lambdas[batch_ind,m+1][:,None])
                        #comb_grad += soft_lam[batch_ind,m+1][:,None]*KD_grads[m]
                        '''if m == 0:
                            comb_grad_all = (1- lambdas_2[m])*SL_grads
                        else:
                            comb_grad_all += (1- lambdas_2[m])*SL_grads'''
                        comb_grad_all += lambdas_2[m]*KD_grads[m]

                    #if r == 0:
                    #    print(KD_grads[m][0][:10])
                    #    print((lambdas[batch_ind,m+1][:,None]*KD_grads[m])[0][:10])

                    comb_grad = comb_grad_all.sum(0)#mean(0)

                    out_vec_val = self.init_out - (eta * comb_grad[:self.num_classes].view(1, -1).\
                        expand(self.init_out.shape[0], -1))

                    out_vec_val = out_vec_val - (eta * torch.matmul(self.init_l1, comb_grad[self.num_classes:].\
                        view(self.num_classes, -1).transpose(0, 1)))

                    #out_vec_val.requires_grad = True
                    loss_SL_val = self.criterion_red(out_vec_val, Y.to(self.device))
                    #self.criterion(out_vec_val, Y.to(self.device)).sum()
                    

                    #if len(self.teacher_model) == 1:
                    '''true_y = torch.zeros_like(tea_out_val)
                    true_y.scatter_(1, Y.to(self.device).data.unsqueeze(1), smoothing)
                    soft_teacher = ((1-smoothing)*F.softmax(tea_out_val/c_temp, dim=1) + true_y)
                    
                    loss_KD_val = c_temp * c_temp *nn.KLDivLoss(reduction='batchmean')(F.log_softmax(\
                    out_vec_val/c_temp , dim=1), soft_teacher)'''

                    """l0_grads = (torch.autograd.grad(loss_SL_val, out_vec_val)[0]).detach().clone().to(self.device)
                    #l0_grads = (torch.autograd.grad(loss_KD_val, out_vec_val)[0]).detach().clone().to(self.device)
                    
                    l0_expand = torch.repeat_interleave(l0_grads, self.init_l1.shape[1], dim=1)
                    l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes).to(self.device)
                    up_grads_val = torch.cat((l0_grads, l1_grads), dim=1).sum(0)

                    #up_grads_val = up_grads_val/torch.norm(up_grads_val)

                    out_vec = train_out - (eta * comb_grad[:self.num_classes].view(1, -1).expand(train_out.shape[0], -1))

                    out_vec = out_vec - (eta * torch.matmul(train_l1, comb_grad[self.num_classes:].\
                        view(self.num_classes, -1).transpose(0, 1)))

                    #out_vec.requires_grad = True

                    #loss_SL_trn = self.criterion_red(out_vec, train_target) #self.smoothLoss(out_vec, train_target)   
                    
                    true_y = torch.zeros_like(tea_out)
                    true_y.scatter_(1, train_target.data.unsqueeze(1), smoothing)
                    soft_teacher = ((1-smoothing)*F.softmax(tea_out/c_temp, dim=1) + true_y)
                    #soft_teacher = F.softmax(tea_out/c_temp, dim=1)
                    loss_KD_trn = c_temp * c_temp *nn.KLDivLoss(reduction='batchmean')(F.log_softmax(\
                    out_vec/c_temp , dim=1), soft_teacher)


                    l0_grads = (torch.autograd.grad(loss_KD_trn, out_vec)[0]).detach().clone().to(self.device)
                    #l0_grads = (torch.autograd.grad(loss_SL_trn, out_vec)[0]).detach().clone().to(self.device)
                    l0_expand = torch.repeat_interleave(l0_grads, train_l1.shape[1], dim=1)
                    l1_grads = l0_expand * train_l1.repeat(1, self.num_classes).to(self.device)
                    up_grads = torch.cat((l0_grads, l1_grads), dim=1).sum(0)"""

                    #up_grads = up_grads/torch.norm(up_grads)
                    #combined = ((up_grads_val+0.4*up_grads)/torch.norm(up_grads_val+0.4*up_grads)).T #+0.1*up_grads
                    #combined = (up_grads_val + 0.4*up_grads).T
                    #combined = (up_grads_val).T

                    #print(torch.matmul(torch.mean(KD_grads[m],dim=0),up_grads).item(),\
                    #    torch.matmul(torch.mean(SL_grads,dim=0),up_grads).item(),end=",")
                    
                    alpha_grads =  (torch.autograd.grad(loss_SL_val, lambdas1,retain_graph=True)[0]).detach().clone().to(self.device)  
                    lam[batch_ind,0] = lam[batch_ind,0] - 1500*alpha_grads.view(-1)

                    for m in range(len(self.teacher_model)):
                        #grad = (-soft_lam[batch_ind,0]*soft_lam[batch_ind,m+1])[:,None]*SL_grads 
                        #grad = KD_grads[m]/torch.norm(KD_grads[m],dim=1)[:,None] -SL_grads/torch.norm(SL_grads,dim=1)[:,None]
                        #grad = KD_grads[m] -SL_grads
                        
                        #alpha_grads = torch.matmul(grad,combined)#/(torch.norm(grad,dim=1)*torch.norm(combined))
                        
                        alpha_grads =  (torch.autograd.grad(loss_SL_val, lambdas_2[m])[0]).detach().clone().to(self.device)  
                        lam[batch_ind,m+1] = lam[batch_ind,m+1] - 1500*alpha_grads.view(-1) #9*eta*"""
                        #lam[batch_ind,0] = lam[batch_ind,0] + 100*alpha_grads.view(-1)

                    if (batch_idx + 1) % (self.fit*40) ==0:
                        '''mean_sl = torch.mean(SL_grads,dim=0)#/torch.norm(torch.mean(SL_grads,dim=0))
                        mean_kd = torch.mean(KD_grads[m],dim=0)#/torch.norm(torch.mean(KD_grads[m],dim=0))
                        up_val_norm =  up_grads_val#/torch.norm(up_grads_val)
                        up_trn_norm =  up_grads#/torch.norm(up_grads)
                        #print(round(loss_SL_val.item(),4),"+",round(loss_KD_val.item(),4),"+",\
                        #round(loss_SL.item(),4),"+",round(loss_KD.item(),4), end=",")
                        if r ==0:
                            print(torch.matmul(mean_sl,mean_kd).item(),torch.matmul(mean_kd,up_val_norm).item(),\
                            torch.matmul(mean_sl,up_val_norm).item(),torch.matmul(mean_kd,up_trn_norm).item(),\
                            torch.matmul(mean_sl,up_trn_norm).item(),end=",")
                            print("Element 0",end=",")
                            print(torch.matmul(SL_grads[0],KD_grads[m][0]).item(),torch.matmul(KD_grads[m][0],up_val_norm).item(),\
                            torch.matmul(SL_grads[0],up_val_norm).item(),torch.matmul(KD_grads[m][0],up_trn_norm).item(),\
                            torch.matmul(SL_grads[0],up_trn_norm).item(),end=",")
                        #print(round(loss_SL_val.item(),4),"+",round(loss_SL_trn.item(),4), end=",")'''
                        if r ==0:
                            print(round(self.criterion_red(self.init_out, Y.to(self.device)).item(),4))
                        print(alpha_grads[0],round(loss_SL_val.item(),4),end=",")#"+",round(loss_KD_trn.item(),4), )
                        #print(self.init_out[r][self.y_val[r]],out_vec_val[r][self.y_val[r]])
                        #print(alpha_grads[:3])
                    del out_vec_val
                    #del out_vec
                    #print("After",lambdas[batch_ind[0]].item(),lambdas[batch_ind[-1]].item())
                    lam.clamp_(min=1e-7,max=max_value-1e-7)
                    #soft_lam = F.softmax(lambdas, dim=1)
                    #lam[batch_ind,0] = max_value- torch.max(lam[batch_ind,1:],dim=1).values
                if (batch_idx + 1) % (self.fit*40) ==0:
                    print()#"End for loop")

        #lambdas.clamp_(min=0.01,max=0.99)
        return lam



