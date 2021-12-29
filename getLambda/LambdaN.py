import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

import faulthandler


class RewLambda(object):
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

    def get_lambdas(self, eta):

        offset = 0
        batch_wise_indices = list(self.trainloader.batch_sampler)

        lambdas = torch.ones(self.N_trn, device=self.device)

        with torch.no_grad():

            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                targets =  targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    count = 0
                    for b in inputs:
                        out, l1 = self.model(b.to(self.device),last=True, freeze=True)
                        if count == 0:
                            full_out = out.cuda(1)
                            full_l1 = l1.cuda(1)
                        else:
                            full_out = torch.cat((full_out,out.cuda(1)), dim=1)
                            full_l1 = torch.cat((full_l1,l1.cuda(1)), dim=1)
                        count+=1

                    re_out = full_out.reshape((full_out.shape[0], -1, out.shape[1]))
                    re_l1  = full_l1.reshape((full_l1.shape[0], -1, l1.shape[1]))
                    _,ind_out = torch.median(re_out, dim=1)
                    #_, ind_out = torch.min(re_out, dim=1)
                    #_,ind_l1 = torch.median(re_l1, dim=1)

                    #del full_out, re_out

                    self.init_out = re_out[range(full_out.shape[0]), ind_out[range(full_out.shape[0]), targets]]
                    self.init_l1 = re_l1[range(full_out.shape[0]), ind_out[range(full_out.shape[0]), targets]]
                    self.y_val = targets.cuda(1)  # .view(-1, 1)
                    #tea_out_val = self.teacher_model(b.to(self.device))
                else:
                    count = 0
                    for b in inputs:
                        out, l1 = self.model(b.to(self.device), last=True, freeze=True)

                        if count == 0:
                            full_out = out.cuda(1)
                            full_l1 = l1.cuda(1)
                        else:
                            full_out = torch.cat((full_out, out.cuda(1)), dim=1)
                            full_l1 = torch.cat((full_l1, l1.cuda(1)), dim=1)
                        count += 1

                    re_out = full_out.reshape((full_out.shape[0], -1, out.shape[1]))
                    re_l1 = full_l1.reshape((full_l1.shape[0], -1, l1.shape[1]))
                    _, ind_out = torch.median(re_out, dim=1)
                    #_, ind_out = torch.min(re_out, dim=1)
                    #_, ind_l1 = torch.median(re_l1, dim=1)

                    #del full_out, re_out
                    #out, l1 = self.model(inputs[ind_out[range(targets.shape[0]),targets]], last=True, freeze=True)

                    #print(re_out[:, ind_out[:, targets]].shape)
                    self.init_out = torch.cat((self.init_out,re_out[range(full_out.shape[0]), ind_out[range(full_out.shape[0]), targets]]), dim=0)
                    self.init_l1 = torch.cat((self.init_l1,re_l1[range(full_out.shape[0]), ind_out[range(full_out.shape[0]), targets]]), dim=0)
                    self.y_val = torch.cat((self.y_val, targets.cuda(1) ), dim=0)
                    #tea_out_val = torch.cat((tea_out_val, self.teacher_model(b.to(self.device))), dim=0)

            # val_loss_SL = self.criterion_red(self.init_out,self.y_val)

            val_loss_SL = torch.sum(self.criterion(self.init_out, self.y_val))

            # print(val_loss_SL)

            '''val_loss_KD = nn.KLDivLoss(reduction='none')(F.log_softmax(self.init_out / self.temp, dim=1),\
                F.softmax(tea_out_val / self.temp, dim=1))
            val_loss_KD = self.temp*self.temp*torch.sum (val_loss_KD)#torch.mean(torch.sum (val_loss_KD,dim=1))'''

            # val_loss_KD = self.temp*self.temp*nn.KLDivLoss(reduction='batchmean')(F.log_softmax(self.init_out / self.temp, dim=1),\
            #    F.softmax(tea_out_val / self.temp, dim=1))

            self.init_out = self.init_out.cuda(1)
            self.init_l1 = self.init_l1.cuda(1)
            self.y_val = self.y_val.cuda(1)
            #tea_out_val = tea_out_val.cuda(1)

        for batch_idx, (inputs, target) in enumerate(self.trainloader):

            batch_wise_indices = list(self.trainloader.batch_sampler)

            target = target.cuda(1)#to(self.device, non_blocking=True)

            count = 0
            for b in inputs:
                part_out, part_l1 = self.model(b.to(self.device), last=True, freeze=True)
                with torch.no_grad():
                    part_teacher_outputs = self.teacher_model(b.to(self.device))

                if count == 0:
                    outputs = part_out.cuda(1)
                    l1 = part_l1.cuda(1)
                    teacher_outputs = part_teacher_outputs.cuda(1)
                else:
                    outputs = torch.cat((outputs, part_out.cuda(1)), dim=1)
                    l1 = torch.cat((l1, part_l1.cuda(1)), dim=1)
                    teacher_outputs = torch.cat((teacher_outputs,part_teacher_outputs.cuda(1)), dim=1)
                count += 1

            oshape = outputs.shape
            re_out = outputs.reshape((outputs.shape[0], -1, part_out.shape[1]))
            re_l1 = l1.reshape((l1.shape[0], -1, part_l1.shape[1]))
            re_tout = teacher_outputs.reshape((teacher_outputs.shape[0], -1, part_out.shape[1]))
            _, ind_out = torch.median(re_out, dim=1)
            #_, ind_out = torch.min(re_out, dim=1)
            #_, ind_l1 = torch.median(re_l1, dim=1)
            #_, ind_tout = torch.median(re_tout, dim=1)

            outputs = re_out[range(oshape[0]), ind_out[range(oshape[0]), target]] #re_out[:,0]
            l1 = re_l1[range(oshape[0]), ind_out[range(oshape[0]), target]] #re_l1[:,0]
            teacher_outputs = re_tout[range(oshape[0]), ind_out[range(oshape[0]), target]] #re_tout[:,0

            loss_SL = self.criterion_red(outputs, target)  # self.criterion(outputs, target).sum()

            l0_grads = (torch.autograd.grad(loss_SL, outputs)[0]).detach().clone().cuda(1)
            l0_expand = torch.repeat_interleave(l0_grads, l1.shape[1], dim=1)
            l1_grads = l0_expand * l1.repeat(1, self.num_classes).cuda(1)

            if batch_idx % self.fit == 0:
                SL_grads = torch.cat((l0_grads, l1_grads), dim=1)
                batch_ind = batch_wise_indices[batch_idx]
            else:
                SL_grads = torch.cat((SL_grads, torch.cat((l0_grads, l1_grads), dim=1)), dim=0)
                batch_ind.extend(batch_wise_indices[batch_idx])

                if (batch_idx + 1) % self.fit == 0 or batch_idx + 1 == len(self.trainloader):
                    out_vec = self.init_out.repeat(SL_grads.shape[0], 1, 1).cuda(1) - (eta * \
                                                                                       SL_grads[:,
                                                                                       :self.num_classes].reshape(
                                                                                           SL_grads.shape[0], 1,
                                                                                           -1).repeat(1,
                                                                                                      self.init_out.shape[
                                                                                                          0], 1))

                    out_vec = out_vec - (eta * torch.matmul(self.init_l1.repeat(SL_grads.shape[0], 1, 1), \
                                                            SL_grads[:, self.num_classes:].reshape(SL_grads.shape[0],
                                                                                                   self.num_classes,
                                                                                                   -1).transpose(1, 2)))

                    up_loss_SL = self.criterion(out_vec.reshape(-1, self.init_out.shape[1]), \
                                                self.y_val.repeat(SL_grads.shape[0]))

                    # re_loss_SL = torch.mean(up_loss_SL.reshape(SL_grads.shape[0],-1),dim=1)
                    re_loss_SL = torch.sum(up_loss_SL.reshape(SL_grads.shape[0], -1), dim=1)
                    # print(re_loss_SL.max(),re_loss_SL.min(),torch.median(re_loss_SL),torch.quantile(re_loss_SL, 0.75),torch.quantile(re_loss_SL, 0.25))
                    diff_loss_SL = val_loss_SL.cuda(1) - re_loss_SL
                    # print(diff_loss_SL.max(),diff_loss_SL.min(),torch.median(diff_loss_SL),torch.quantile(diff_loss_SL, 0.75),torch.quantile(diff_loss_SL, 0.25))

            '''loss_KD = nn.KLDivLoss(reduction='none')(F.log_softmax(outputs / self.temp, dim=1),\
                F.softmax(teacher_outputs / self.temp, dim=1))
            loss_KD = self.temp*self.temp*torch.sum(loss_KD)'''

            loss_KD = self.temp * self.temp * nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(outputs / self.temp, dim=1), \
                F.softmax(teacher_outputs / self.temp, dim=1))

            l0_grads = (torch.autograd.grad(loss_KD, outputs)[0]).detach().clone().cuda(1)
            l0_expand = torch.repeat_interleave(l0_grads, l1.shape[1], dim=1)
            l1_grads = l0_expand * l1.repeat(1, self.num_classes).cuda(1)

            if batch_idx % self.fit == 0:
                KD_grads = torch.cat((l0_grads, l1_grads), dim=1)
            else:
                KD_grads = torch.cat((KD_grads, torch.cat((l0_grads, l1_grads), dim=1)), dim=0)

                if (batch_idx + 1) % self.fit == 0 or batch_idx + 1 == len(self.trainloader):
                    out_vec = self.init_out.repeat(KD_grads.shape[0], 1, 1) - (eta * \
                                                                               KD_grads[:, :self.num_classes].reshape(
                                                                                   KD_grads.shape[0], 1, -1).repeat(1,
                                                                                                                    self.init_out.shape[
                                                                                                                        0],
                                                                                                                    1))

                    out_vec = out_vec - (eta * torch.matmul(self.init_l1.repeat(KD_grads.shape[0], 1, 1), \
                                                            KD_grads[:, self.num_classes:].reshape(KD_grads.shape[0],
                                                                                                   self.num_classes,
                                                                                                   -1).transpose(1, 2)))

                    '''up_loss_KD = nn.KLDivLoss(reduction='none')\
                        (F.log_softmax(out_vec.reshape(-1,self.init_out.shape[1]) / self.temp, dim=1),\
                        F.softmax(tea_out_val.repeat(KD_grads.shape[0],1) / self.temp, dim=1))
                    up_loss_KD = self.temp*self.temp*torch.sum (up_loss_KD,dim=1)
                    diff_loss_KD = val_loss_KD.cuda(1) - torch.mean(up_loss_KD.reshape(KD_grads.shape[0],-1),dim=1)'''

                    up_loss_KD = self.criterion(out_vec.reshape(-1, self.init_out.shape[1]), \
                                                self.y_val.repeat(KD_grads.shape[0]))

                    # re_loss_KD = torch.mean(up_loss_KD.reshape(KD_grads.shape[0],-1),dim=1)
                    re_loss_KD = torch.sum(up_loss_KD.reshape(KD_grads.shape[0], -1), dim=1)
                    # print(re_loss_KD.max(),re_loss_KD.min(),torch.median(re_loss_KD),torch.quantile(re_loss_KD, 0.75),torch.quantile(re_loss_KD, 0.25))
                    diff_loss_KD = val_loss_SL.cuda(1) - re_loss_KD
                    # print(diff_loss_KD.max(),diff_loss_KD.min(),torch.median(diff_loss_KD),torch.quantile(diff_loss_KD, 0.75),torch.quantile(diff_loss_KD, 0.25))

                    """diff_loss_SL[diff_loss_SL == 0 ] = 0.1
                    diff_loss_KD[diff_loss_KD == 0 ] = 0.1
                    diff_loss_KD[diff_loss_KD <0 ] = 0
                    diff_loss_SL[diff_loss_SL < 0] = 0
 
                    sum_loss = diff_loss_KD+diff_loss_SL
                    corr_lam = diff_loss_KD/sum_loss
                    corr_lam[sum_loss == 0] = (val_loss_SL.cuda(1) - re_loss_SL[sum_loss == 0])/(val_loss_SL.cuda(1) - re_loss_KD[sum_loss == 0] + val_loss_SL.cuda(1) - re_loss_SL[sum_loss == 0])"""

                    '''tem = torch.isnan(corr_lam)
                    corr_lam[torch.where(tem == True)[0]] = 0.5'''

                    combine = torch.stack((diff_loss_SL, diff_loss_KD), dim=1)
                    combine = combine* (10**(-torch.floor(torch.log10(torch.median(torch.abs(combine))))))

                    #print(combine[0],combine[10],combine[-10],combine[-1])
                    '''mini = torch.min(combine, dim=1)[0]
                    # print(mini)
                    # print(torch.min(torch.abs(mini[mini != 0]))/10)

                    corr_com = combine - mini[:, None] + torch.min(torch.abs(mini[mini != 0]))  # /10
                    corr_lam = (corr_com[:, 1]) / (torch.sum(corr_com, dim=1))
                    lambdas[batch_ind] = corr_lam.cuda(0)'''

                    frac = F.softmax(combine, dim=1)
                    #print(frac[0], frac[10], frac[-10], frac[-1])
                    lambdas[batch_ind] = frac[:,1].cuda(0)

        return lambdas


"""min_SL = torch.min(diff_loss_SL)
min_KD = torch.min(diff_loss_KD)

min_min = torch.min(min_SL,min_KD)

if min_min < 1e-7:
   corr_loss_SL = diff_loss_SL - min_min + 1e-7
else:
   corr_loss_SL = diff_loss_SL

if min_min < 1e-7:
   corr_loss_KD = diff_loss_KD - min_min + 1e-7
else:
   corr_loss_SL = diff_loss_SL

sum_loss = corr_loss_KD+corr_loss_SL
corr_lam = corr_loss_KD/sum_loss

'''corr_lam[corr_lam < torch.quantile(corr_lam, 0.1)] = torch.quantile(corr_lam, 0.1)
corr_lam[corr_lam > torch.quantile(corr_lam, 0.9)] = torch.quantile(corr_lam, 0.9)'''

top_1 = torch.quantile(corr_lam, 0.99)

fix_lam = corr_lam.clone()

fix_lam[corr_lam > 0.5 ] = torch.clamp(0.5 + 0.5*(corr_lam[corr_lam > 0.5 ])/(top_1),max=1.0)
#fix_lam[corr_lam < 0.5] = 0.5*(corr_lam[corr_lam < 0.5])/(corr_lam[corr_lam < 0.5].max())

lambdas[batch_ind] =  fix_lam.cuda(0)"""

'''SL_zero_ind = torch.zeros_like(mean_loss_SL, dtype=torch.bool)
SL_zero_ind[torch.where(mean_loss_SL==0)[0]] = True
KD_zero_ind = torch.zeros_like(mean_loss_KD, dtype=torch.bool)
KD_zero_ind[torch.where(mean_loss_KD==0)[0]] = True

SL_min = torch.min(mean_loss_SL[~SL_zero_ind])
KD_min = torch.min(mean_loss_KD[~KD_zero_ind])

corr_lam[SL_zero_ind] = mean_loss_KD[SL_zero_ind]/(mean_loss_KD[SL_zero_ind]+SL_min)
corr_lam[KD_zero_ind] = KD_min/(mean_loss_SL[KD_zero_ind]+KD_min)
corr_lam[torch.where(sum_loss == 0)[0]] = 0.5

tem = torch.isnan(corr_lam)
if len(torch.where(tem == True)[0]) > 0:
   print(corr_loss_KD[torch.where(tem == True)[0]],corr_loss_SL[torch.where(tem == True)[0]])'''

'''diff = re_loss_SL - re_loss_KD
#print(diff.max(),diff.min(),torch.median(diff),torch.quantile(diff, 0.75),torch.quantile(diff, 0.25))
corr_lam  = diff.clone()
corr_lam[diff < torch.quantile(diff, 0.1)] = torch.quantile(diff, 0.1)
corr_lam[diff > torch.quantile(diff, 0.9)] = torch.quantile(diff, 0.9)
#print(corr_lam.max(),corr_lam.min(),torch.median(corr_lam),torch.quantile(corr_lam, 0.75),torch.quantile(corr_lam, 0.25))


if len(corr_lam[diff >= 0]) > 0:
   corr_lam[diff >= 0] = 0.5 + 0.1*(corr_lam[diff >= 0])/(corr_lam[diff >= 0].max())
if len(corr_lam[diff < 0]) > 0:
   corr_lam[diff < 0] = 0.5*(corr_lam[diff < 0] - corr_lam[diff < 0].min())/( - corr_lam[diff < 0].min())

if len(corr_lam[diff_loss_SL > 0]) > 0:
   corr_lam[diff_loss_SL > 0] = 0.1*(diff_loss_SL[diff_loss_SL > 0])/(diff_loss_SL[diff_loss_SL >= 0].max())

if len(corr_lam[diff_loss_KD > 0]) > 0:
   corr_lam[diff_loss_KD > 0] = 0.9 + 0.1*(diff_loss_KD[diff_loss_KD > 0])/(diff_loss_KD[diff_loss_KD > 0].max())'''

# lambdas[batch_ind] =  corr_lam.cuda(0)


