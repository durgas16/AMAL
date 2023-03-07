import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

class LearnMultiLambdaMeta(object):
    
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
        self.temp = temp

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
        
        KD_grads = [0 for _ in range(len(self.teacher_model))]

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
                
                for r in range(5):

                    lambdas = lam.clone().to(self.device)#, device=self.device)
                    lambdas.requires_grad = True
                     
                    lambdas1 = lambdas[batch_ind,0][:,None]
                    comb_grad_all = lambdas1*SL_grads 
                    
                    lambdas_2 =[]
                    for m in range(len(self.teacher_model)):
                        lambdas_2.append(lambdas[batch_ind,m+1][:,None])
                        comb_grad_all += lambdas_2[m]*KD_grads[m]

                    comb_grad = comb_grad_all.sum(0)#mean(0)

                    out_vec_val = self.init_out - (eta * comb_grad[:self.num_classes].view(1, -1).\
                        expand(self.init_out.shape[0], -1))

                    out_vec_val = out_vec_val - (eta * torch.matmul(self.init_l1, comb_grad[self.num_classes:].\
                        view(self.num_classes, -1).transpose(0, 1)))

                    #out_vec_val.requires_grad = True
                    loss_SL_val = self.criterion_red(out_vec_val, Y.to(self.device))
                    #self.criterion(out_vec_val, Y.to(self.device)).sum()
                    
                    alpha_grads =  (torch.autograd.grad(loss_SL_val, lambdas1,retain_graph=True)[0]).detach().clone().to(self.device)  
                    lam[batch_ind,0] = lam[batch_ind,0] - 1500*alpha_grads.view(-1)

                    for m in range(len(self.teacher_model)):
                        
                        alpha_grads =  (torch.autograd.grad(loss_SL_val, lambdas_2[m])[0]).detach().clone().to(self.device)  
                        lam[batch_ind,m+1] = lam[batch_ind,m+1] - 1500*alpha_grads.view(-1) 

                    del out_vec_val

                    lam.clamp_(min=1e-7,max=max_value-1e-7)
                   
        #lambdas.clamp_(min=0.01,max=0.99)
        return lam



