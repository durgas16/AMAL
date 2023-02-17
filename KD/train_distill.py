
import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
import os.path as osp

import torch.nn.functional as F

from utils.custom_dataset import load_dataset_custom
from utils.config_utils import load_config_data
from utils.scheduler import Scheduler

from models import *

from .LearnMultiLambdaMeta import LearnMultiLambdaMeta

import argparse

parser = argparse.ArgumentParser(description='Adaptive Mixing of losses between 0 to 1')
# General
parser.add_argument('--lam', default=0.9, type=float,help='loss mixing parameter')  

parser.add_argument('--temp', default=4, type=int,help='temperature for KD loss')                        

parser.add_argument('--seed', type=int, default=-1, help='random seed, set as -1 for random.')

parser.add_argument('--con_file', type=str, default="config/cifar100_wrn/config_multilam_cifar100.py",\
     help='path to the config file')


args = parser.parse_args()
args_dict = args.__dict__
args = argparse.Namespace(**args_dict)

seed = args.seed 
torch.manual_seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

Temp = args.temp 
_lambda = args.lam

print(seed,_lambda)

class MyDataset(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]

        return data, target, index

    def __len__(self):
        return len(self.dataset)


class TrainClassifier:
    def __init__(self, config_file):
        self.config_file = config_file
        self.configdata = load_config_data(self.config_file)

    """
    Loss Evaluation
    """

    def model_eval_loss(self, data_loader, model, criterion):
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                    self.configdata['train_args']['device'], non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss

    """
    #Model Creation
    """

    def create_model(self, mtype,hid_unit=None,d=None,w=None):

        if mtype == 'NN_2L':
            print(mtype , hid_unit)
        elif mtype in ['WRN_16_X','DenseNet_X']:
            print(mtype , d, w )
        elif mtype in ['CNN_X']:
            print(mtype , d )
        else:
            print(mtype)

        if mtype == 'ResNet8':
            if self.configdata['dataset']['name'] in ['airplane','cars']:
                model = resnet8_cifar(num_classes=self.configdata['model']['numclasses'],if_large=True)
            else:
                model = resnet8_cifar(num_classes=self.configdata['model']['numclasses'])
        elif mtype == 'ResNet14':
            model = resnet14_cifar(num_classes=self.configdata['model']['numclasses'])
        elif mtype == 'ResNet20':
            model = resnet20_cifar(num_classes=self.configdata['model']['numclasses'])
        elif mtype == 'ResNet56':
            model = resnet56_cifar(num_classes=self.configdata['model']['numclasses'])
        elif mtype == 'ResNet110':
            model = resnet110_cifar(num_classes=self.configdata['model']['numclasses'])
        elif mtype == 'ResNet32':
            model = resnet32_cifar(num_classes=self.configdata['model']['numclasses'])
        elif mtype == 'WRN_16_X':
            if self.configdata['dataset']['name'] in ['cars','airplane']:
                model = WRN_16_X(depth=d, width =w,num_classes=self.configdata['model']['numclasses'],if_large=True)
            else:
                model = WRN_16_X(depth=d, width =w,num_classes=self.configdata['model']['numclasses'],if_large=False)
        
        elif mtype == 'DenseNet_X':
            model = DN_X_Y(depth=d, g =w,num_classes=self.configdata['model']['numclasses'])
            
        elif mtype == 'CNN_X':
            model = create_cnn_model(d, num_classes=self.configdata['model']['numclasses'])
        elif mtype == 'NN_2L':
            model = TwoLayerNet(input_dim=self.configdata['model']['input_dims'], \
                                num_classes=self.configdata['model']['numclasses'], \
                                hidden_units=hid_unit)

        model = model.to(self.configdata['train_args']['device'])
        return model

    """#Loss Type, Optimizer and Learning Rate Scheduler"""

    def loss_function(self):
        if self.configdata['loss']['type'] == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
            criterion_nored = nn.CrossEntropyLoss(reduction='none')
        return criterion, criterion_nored

    def optimizer_with_scheduler(self, elements,if_model=True):

        if if_model:
            leaves = elements.parameters()
        else:
            leaves = elements
        if self.configdata['optimizer']['type'] == 'sgd':
            if self.configdata['model']['architecture'] == 'CNN_X':
                optimizer = optim.SGD(leaves , lr=self.configdata['optimizer']['lr'],
                                  momentum=self.configdata['optimizer']['momentum'],
                                  weight_decay=self.configdata['optimizer']['weight_decay'], nesterov=False)
            else:
                optimizer = optim.SGD(leaves , lr=self.configdata['optimizer']['lr'],
                                    momentum=self.configdata['optimizer']['momentum'],
                                    weight_decay=self.configdata['optimizer']['weight_decay'], nesterov=True)
        elif self.configdata['optimizer']['type'] == "adam":
            optimizer = optim.Adam(leaves , lr=self.configdata['optimizer']['lr'])
        elif self.configdata['optimizer']['type'] == "rmsprop":
            optimizer = optim.RMSprop(leaves , lr=self.configdata['optimizer']['lr'])

        if self.configdata['scheduler']['type'] == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.configdata['scheduler'][
                'T_max'])  # ,eta_min=8e-4
        elif self.configdata['scheduler']['type'] == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120],
                                                             gamma=0.1)
        elif self.configdata['scheduler']['type'] == 'Mstep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,180,210], gamma=0.1)                                                       
        elif self.configdata['scheduler']['type'] == 'RPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', \
                                                                   factor=self.configdata['scheduler']['decay'],
                                                                   patience=10, threshold=0.0001, \
                                                                   threshold_mode='rel', cooldown=5, min_lr=1e-08,
                                                                   eps=1e-07, verbose=False)
        return optimizer, scheduler


    def save_ckpt(self, state, ckpt_path):
        torch.save(state, ckpt_path)

    def load_ckp(self, ckpt_path, model, optimizer, scheduler): #ema_model,
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        #ema_model.load_state_dict(checkpoint['state_dict_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        loss = checkpoint['loss']
        metrics = checkpoint['metrics']
        
        all_outputs = checkpoint['output']

        if self.configdata['ds_strategy']['type'] in ['MultiLam']:
            all_lambda = checkpoint['lambda']
            return start_epoch, model, optimizer, scheduler, loss, metrics, all_lambda,all_outputs #ema_model,
        else:
            return start_epoch, model, optimizer, scheduler, loss, metrics, all_outputs

    def train(self):
        """
        #General Training Loop with Data Selection Strategies
        """
        
        if self.configdata['dataset']['name'] == "synthetic":
            trainset, validset, testset, num_cls = load_dataset_custom(self.configdata['dataset']['datadir'],
                                                                        self.configdata['dataset']['name'],
                                                                        self.configdata['dataset']['feature'],
                                                                        seed=42,
                                                                        valid=True,
                                                                        type=self.configdata['dataset']['type'], \
                                                                        test_type=self.configdata['dataset'][
                                                                            'test_type'], \
                                                                        ncls=self.configdata['model']['numclasses']) 
        else:
            trainset, validset, testset, num_cls = load_dataset_custom(self.configdata['dataset']['datadir'],
                                                                        self.configdata['dataset']['name'],
                                                                        self.configdata['dataset']['feature'],
                                                                        valid=True)

        N = len(trainset)
        trn_batch_size = self.configdata['dataloader']['trn_batch_size']
        val_batch_size = self.configdata['dataloader']['val_batch_size']
        tst_batch_size = self.configdata['dataloader']['tst_batch_size']

        # Creating the Data Loaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                                  shuffle=False, pin_memory=True)


        trainloader_ind = torch.utils.data.DataLoader(MyDataset(trainset), batch_size=trn_batch_size,
                                                  shuffle=True, pin_memory=True)

        valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size,
                                                shuffle=True, pin_memory=True)


        testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
                                                 shuffle=False, pin_memory=True)

        trn_losses = list()
        val_losses = list()  
        tst_losses = list()
 
        trn_acc = list()
        val_acc = list()  
        tst_acc = list()  

        
        # Results logging file
        print_every = self.configdata['train_args']['print_every']
        results_dir = osp.abspath(osp.expanduser(self.configdata['train_args']['results_dir']))
        if self.configdata['dataset']['name'] == "synthetic":
            if self.configdata['model']['architecture'] == 'NN_2L':
                all_logs_dir = os.path.join(results_dir, self.configdata['ds_strategy']['type'] + "_distilT",
                                            self.configdata['dataset']['name'],
                                            self.configdata['model']['architecture'] + "_" + str(self.configdata['model']['hidden_units_stu'])\
                                            + "_p" + str(_lambda * 10), \
                                            self.configdata['dataset']['type'],
                                            self.configdata['dataset']['test_type'],
                                            str(seed))
            else:
                all_logs_dir = os.path.join(results_dir, self.configdata['ds_strategy']['type'] + "_distilT",
                                            self.configdata['dataset']['name'],
                                            self.configdata['model']['architecture']+ "_p" + str(_lambda * 10), \
                                            self.configdata['dataset']['type'], self.configdata['dataset']['test_type'],
                                            str(seed))
        else:
            if self.configdata['model']['architecture'] in ['WRN_16_X','DenseNet_X']:
                if _lambda > 0:
                    all_logs_dir = os.path.join(results_dir, self.configdata['ds_strategy']['type'] + "_distilT",\
                    self.configdata['dataset']['name'],self.configdata['model']['architecture']+"_"+\
                    str(self.configdata['model']['depth_teach'])+"_"+str(self.configdata['model']['width_teach']) +"_"+\
                    str(self.configdata['model']['depth'])+ "_"+str(self.configdata['model']['width'])+"_p" + \
                    str(_lambda * 10),str(Temp),str(self.configdata['ds_strategy']['select_every']),str(seed))
                else:
                    all_logs_dir = os.path.join(results_dir, self.configdata['ds_strategy']['type'] + "_distilT",\
                    self.configdata['dataset']['name'],self.configdata['model']['architecture']+"_"+\
                    str(self.configdata['model']['depth'])+ "_"+str(self.configdata['model']['width'])+"_p" + \
                    str(_lambda * 10),str(Temp),str(self.configdata['ds_strategy']['select_every']),str(seed))
            elif self.configdata['model']['architecture'] in ['CNN_X']:
                if _lambda > 0:
                    all_logs_dir = os.path.join(results_dir, self.configdata['ds_strategy']['type'] + "_distilT",\
                    self.configdata['dataset']['name'],self.configdata['model']['architecture']+"_"+\
                    str(self.configdata['model']['depth_teach'])+"_"+str(self.configdata['model']['depth'])+ \
                    "_p" + str(_lambda * 10),str(seed))
                else:
                    all_logs_dir = os.path.join(results_dir, self.configdata['ds_strategy']['type'] + "_distilT",\
                    self.configdata['dataset']['name'],self.configdata['model']['architecture']+"_"+\
                    str(self.configdata['model']['depth'])+"_p" + str(_lambda * 10),str(seed))
            else:
                all_logs_dir = os.path.join(results_dir, self.configdata['ds_strategy']['type'] + "_distilT",
                                            self.configdata['dataset']['name'],
                                            self.configdata['model']['architecture'] + "_p" + str(_lambda * 10),\
                                            str(Temp),str(self.configdata['ds_strategy']['select_every']),str(seed))
        os.makedirs(all_logs_dir, exist_ok=True)
        path_logfile = os.path.join(all_logs_dir, self.configdata['dataset']['name'] + '.txt')
        logfile = open(path_logfile, 'w')

        checkpoint_dir = osp.abspath(osp.expanduser(self.configdata['ckpt']['dir']))
        if self.configdata['dataset']['name'] == "synthetic":
            if self.configdata['model']['architecture'] == 'NN_2L':
                ckpt_dir = os.path.join(checkpoint_dir, self.configdata['ds_strategy']['type'] + "_distilT",\
                self.configdata['dataset']['name'],self.configdata['model']['architecture'] + "_" + \
                str(self.configdata['model']['hidden_units_stu'])+ "_p" + str(_lambda * 10), \
                self.configdata['dataset']['type'],self.configdata['dataset']['test_type'],str(seed))
            else:
                ckpt_dir = os.path.join(checkpoint_dir, self.configdata['ds_strategy']['type'] + "_distilT",
                                            self.configdata['dataset']['name'],
                                            self.configdata['model']['architecture']+ "_p" + str(_lambda * 10), \
                                            self.configdata['dataset']['type'], self.configdata['dataset']['test_type'],
                                            str(seed))
        else:
            if self.configdata['model']['architecture'] in ['WRN_16_X','DenseNet_X']:
                if _lambda > 0:
                    ckpt_dir = os.path.join(checkpoint_dir, self.configdata['ds_strategy']['type'] + "_distilT",\
                    self.configdata['dataset']['name'],self.configdata['model']['architecture']+"_"+\
                    str(self.configdata['model']['depth_teach'])+"_"+str(self.configdata['model']['width_teach']) +"_"+\
                    str(self.configdata['model']['depth'])+ "_"+str(self.configdata['model']['width'])+"_p" + \
                    str(_lambda * 10),str(Temp),str(self.configdata['ds_strategy']['select_every']),str(seed))
                else:
                    ckpt_dir = os.path.join(checkpoint_dir, self.configdata['ds_strategy']['type'] + "_distilT",\
                    self.configdata['dataset']['name'],self.configdata['model']['architecture']+"_"+\
                    str(self.configdata['model']['depth'])+ "_"+str(self.configdata['model']['width'])+"_p" + \
                    str(_lambda * 10),str(Temp),str(self.configdata['ds_strategy']['select_every']),str(seed))

            elif self.configdata['model']['architecture'] in ['CNN_X']:
                if _lambda > 0:
                    ckpt_dir = os.path.join(checkpoint_dir, self.configdata['ds_strategy']['type'] + "_distilT",\
                    self.configdata['dataset']['name'],self.configdata['model']['architecture']+"_"+\
                    str(self.configdata['model']['depth_teach'])+"_"+str(self.configdata['model']['depth'])+ \
                    "_p" + str(_lambda * 10),str(Temp),str(self.configdata['ds_strategy']['select_every']),str(seed))
                else:
                    ckpt_dir = os.path.join(checkpoint_dir, self.configdata['ds_strategy']['type'] + "_distilT",\
                    self.configdata['dataset']['name'],self.configdata['model']['architecture']+"_"+\
                    str(self.configdata['model']['depth'])+"_p" + str(_lambda * 10),\
                    str(Temp),str(self.configdata['ds_strategy']['select_every']),str(seed))
            else:
                ckpt_dir = os.path.join(checkpoint_dir, self.configdata['ds_strategy']['type'] + "_distilT",
                                        self.configdata['dataset']['name'],
                                        self.configdata['model']['architecture'] + "_p" + str(_lambda * 10),\
                                        str(Temp),str(self.configdata['ds_strategy']['select_every']),str(seed))

        checkpoint_path = os.path.join(ckpt_dir, 'model.pt')
        os.makedirs(ckpt_dir, exist_ok=True)

        torch.manual_seed(seed)

        # Model Creation
        
        mtype = self.configdata['model']['architecture']
        hid_unit=None
        d=None
        w=None
        if mtype == 'NN_2L':
            hid_unit = self.configdata['model']['hidden_units_stu']
        elif mtype in ['WRN_16_X','DenseNet_X']:
            d = self.configdata['model']['depth']
            w = self.configdata['model']['width']
        elif mtype in ['CNN_X']:
            d = self.configdata['model']['depth']
        train_model = self.create_model(mtype,hid_unit,d,w)
       
        Nteacher = 1
        if _lambda > 0:
            mtype = self.configdata['model']['teacher_arch']
            hid_unit=None
            d=None
            w=None
            if 'NN_2L' in mtype:
                hid_unit = self.configdata['model']['hidden_units_teach']
            elif 'WRN_16_X' in mtype or 'DenseNet_X' in mtype:
                d = self.configdata['model']['depth_teach']
                w = self.configdata['model']['width_teach']
            elif 'CNN_X' in mtype:
                d = self.configdata['model']['depth_teach']
            
            teacher_model = []
            Nteacher = len(mtype)
            
            for m in range(len(mtype)):

                if mtype[m] == 'NN_2L':
                   teacher_model.append(self.create_model(mtype[m],hid_unit=hid_unit[m]))
                elif mtype[m] in ['WRN_16_X','DenseNet_X']:
                    teacher_model.append(self.create_model(mtype[m],d=d[m],w=w[m]))
                elif mtype[m] in ['CNN_X']:
                    teacher_model.append(self.create_model(mtype[m],d=d[m]))
                else:
                    teacher_model.append(self.create_model(mtype[m]))
                
                print("Teacher",sum(p.numel() for p in teacher_model[-1].parameters() if p.requires_grad))
                print("Loading from",self.configdata['model']['teacher_path'][m])
                checkpoint = torch.load(self.configdata['model']['teacher_path'][m])
                teacher_model[m].load_state_dict(checkpoint['state_dict'])
                #teacher_model.append(copy.deepcopy(model))

        # Loss Functions
        criterion, criterion_nored = self.loss_function()

        # Getting the optimizer and scheduler
        optimizer, scheduler = self.optimizer_with_scheduler(train_model)
        
        if _lambda > 0:
            lambdas = torch.full((N,Nteacher+1),_lambda, device=self.configdata['train_args']['device'])
            
            lambdas[:,0] = 1 - torch.max(lambdas[:,1:],dim=1).values
                
            for m in range(Nteacher+1):
                print(lambdas[:,m].max(), lambdas[:,m].min(), torch.median(lambdas[:,m]),\
                        torch.quantile(lambdas[:,m], 0.75),torch.quantile(lambdas[:,m], 0.25))
        all_outputs =[]
        print("=======================================", file=logfile)

        if self.configdata['ckpt']['is_load'] == True:
            if self.configdata['ds_strategy']['type'] in ['MultiLam']:
                start_epoch, train_model, optimizer, scheduler, ckpt_loss, load_metrics, all_lambda,all_outputs= self.load_ckp(
                    checkpoint_path, train_model, optimizer, scheduler) #ema_model,
                lambdas = copy.deepcopy(all_lambda[-1])
            else:
                start_epoch, train_model, optimizer, scheduler, ckpt_loss, load_metrics, all_outputs= self.load_ckp(
                    checkpoint_path, train_model, optimizer, scheduler) 

            print("Loading saved checkpoint model at epoch " + str(start_epoch))
            
            #soft_lam = F.softmax(lambdas, dim=1)
            for arg in load_metrics.keys():
                
                if arg == "val_loss":
                    val_losses = load_metrics['val_loss']
                if arg == "val_acc":
                    val_acc = load_metrics['val_acc']
                if arg == "tst_loss":
                    tst_losses = load_metrics['tst_loss']
                if arg == "tst_acc":
                    tst_acc = load_metrics['tst_acc']
                if arg == "trn_loss":
                    trn_losses = load_metrics['trn_loss']
                if arg == "trn_acc":
                    trn_acc = load_metrics['trn_acc']

            if self.configdata['ds_strategy']['type'] not in ['No-curr']:
                select = self.configdata['ds_strategy']['select_every']
        else:
            start_epoch = 0
        
        if self.configdata['ds_strategy']['type'] in ['MultiLam']:
            lelam = LearnMultiLambdaMeta(trainloader_ind, valloader, copy.deepcopy(train_model), num_cls, N, \
            criterion_nored, self.configdata['train_args']['device'], self.configdata['dataset']['grad_fit'], \
            teacher_model, criterion, Temp)
            
            select = self.configdata['ds_strategy']['select_every']
            
        print_args = self.configdata['train_args']['print_args']
        for i in range(start_epoch, self.configdata['train_args']['num_epochs']):

            if (self.configdata['ds_strategy']['type'] in ['MultiLam'] and i % select == 0 and i>=select ):

                cached_state_dictT = copy.deepcopy(train_model.state_dict())
                clone_dict = copy.deepcopy(train_model.state_dict())
                lelam.update_model(clone_dict)
                train_model.load_state_dict(cached_state_dictT)

                lambdas = lelam.get_lambdas(optimizer.param_groups[0]['lr'],i,lambdas)
                
                for m in range(Nteacher+1):
                    print(lambdas[:,m].max(), lambdas[:,m].min(), torch.median(lambdas[:,m]),\
                            torch.quantile(lambdas[:,m], 0.75),torch.quantile(lambdas[:,m], 0.25))
                
            train_model.train()
            for batch_idx, (inputs, targets,indices) in enumerate(trainloader_ind):
                inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                    self.configdata['train_args']['device'],
                    non_blocking=True)  # targets can have non_blocking=True.
                optimizer.zero_grad()
                outputs = train_model(inputs)

                if _lambda > 0:                            
                    
                    loss_SL = criterion_nored(outputs, targets)
                    loss = lambdas[indices,0]*loss_SL
                    
                    for m in range(Nteacher):
                        if self.configdata['ds_strategy']['type'] in ['DGKD']:
                            prob = np.random.random(1)[0]
                            if prob >= 0.75:
                                continue
                        with torch.no_grad():
                            teacher_outputs = teacher_model[m](inputs)
                        loss_KD = nn.KLDivLoss(reduction='none')(
                            F.log_softmax(outputs / Temp, dim=1), F.softmax(teacher_outputs / Temp, dim=1))
                        loss +=  Temp * Temp *lambdas[indices,m+1]*torch.sum(loss_KD, dim=1)
                        
                    loss = torch.mean(loss)
                else:
                    loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                del inputs, targets
        
            #print("Training with lr", round(optimizer.param_groups[0]['lr'], 5))
            if self.configdata['scheduler']['type'] == 'RPlateau':    
                scheduler.step(val_losses[-1])
            else:
                scheduler.step()

            trn_loss = 0
            trn_correct = 0
            trn_total = 0
            val_loss = 0
            val_correct = 0
            val_total = 0
            tst_correct = 0
            tst_total = 0
            tst_loss = 0
            train_model.eval()

            curr_outputs = torch.zeros(N,num_cls)
            with torch.no_grad():
                for batch_idx, (inputs, targets,indices) in enumerate(trainloader_ind):
                    
                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                        self.configdata['train_args']['device'], non_blocking=True)
                    outputs = train_model(inputs)
                    curr_outputs[indices] = outputs.cpu()

                    loss = criterion(outputs, targets)
                    trn_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    trn_total += targets.size(0)
                    trn_correct += predicted.eq(targets).sum().item()
                trn_losses.append(trn_loss)
                trn_acc.append(trn_correct / trn_total)
            
            all_outputs.append(curr_outputs)

            if "val_loss" in print_args:
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(valloader):
                        
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                            self.configdata['train_args']['device'], non_blocking=True)
                        outputs = train_model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += targets.size(0)*loss.item()
                        val_total += targets.size(0)
                        
                        if "val_acc" in print_args:
                            _, predicted = outputs.max(1)
                            val_correct += predicted.eq(targets).sum().item()
                    val_losses.append(val_loss / val_total)
                    val_acc.append(val_correct / val_total)

            if "tst_loss" in print_args:
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(testloader):
                        
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                            self.configdata['train_args']['device'], non_blocking=True)
                        outputs = train_model(inputs)
                        loss = criterion(outputs, targets)
                        tst_loss += targets.size(0)*loss.item()
                        tst_total += targets.size(0)
                        if "tst_acc" in print_args:
                            _, predicted = outputs.max(1)
                            tst_correct += predicted.eq(targets).sum().item()

                        if i + 1 == self.configdata['train_args']['num_epochs'] or \
                                ((i + 1) % self.configdata['train_args']['print_every'] == 0):

                            if batch_idx == 0:
                                full_predict = predicted
                                full_logit = outputs
                                full_targets = targets
                            else:
                                full_predict = torch.cat((full_predict, predicted), dim=0)
                                full_logit = torch.cat((full_logit, outputs), dim=0)
                                full_targets = torch.cat((full_targets, targets), dim=0)

                    tst_losses.append(tst_loss/ tst_total)
                    tst_acc.append(tst_correct / tst_total)
                    

            if self.configdata['ds_strategy']['type'] in ['MultiLam'] and (i % select == 0 or i in self.configdata['ds_strategy']['schedule']):
                if (i < self.configdata['ds_strategy']['warm_epoch']):
                    all_lambda = []
                
                all_lambda.append(copy.deepcopy(lambdas))

            if ((i + 1) % self.configdata['train_args']['print_every'] == 0):

                print_str = "Epoch: " + str(i + 1)

                for arg in print_args:

                    if arg == "val_loss":
                        print_str += " , " + "Validation Loss: " + str(val_losses[-1])

                    if arg == "val_acc":
                        print_str += " , " + "Validation Accuracy: " + str(val_acc[-1])

                    if arg == "tst_loss":
                        print_str += " , " + "Test Loss: " + str(tst_losses[-1])

                    if arg == "tst_acc":
                        print_str += " , " + "Test Accuracy: " + str(tst_acc[-1])

                    if arg == "trn_loss":
                        print_str += " , " + "Training Loss: " + str(trn_losses[-1])

                    if arg == "trn_acc":
                        print_str += " , " + "Training Accuracy: " + str(trn_acc[-1])

                print(print_str)

            if ((i + 1) % self.configdata['ckpt']['save_every'] == 0):

                if self.configdata['ckpt']['is_save'] == True:

                    metric_dict = {}

                    for arg in print_args:

                        if arg == "val_loss":
                            metric_dict['val_loss'] = val_losses
                        if arg == "val_acc":
                            metric_dict['val_acc'] = val_acc
                        if arg == "tst_loss":
                            metric_dict['tst_loss'] = tst_losses
                        if arg == "tst_acc":
                            metric_dict['tst_acc'] = tst_acc
                        if arg == "trn_loss":
                            metric_dict['trn_loss'] = trn_losses
                        if arg == "trn_acc":
                            metric_dict['trn_acc'] = trn_acc

                    ckpt_state = {
                        'epoch': i + 1,
                        'state_dict': train_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': self.loss_function(),
                        'metrics': metric_dict,
                        'output': all_outputs
                    }

                    if self.configdata['ds_strategy']['type'] in ['MultiLam']:
                        ckpt_state['lambda']= all_lambda

                    # save checkpoint
                    self.save_ckpt(ckpt_state, checkpoint_path)
                    print("Model checkpoint saved at epoch " + str(i + 1))

        print(self.configdata['ds_strategy']['type'] + " Selection Run---------------------------------")
        if "val_loss" in print_args:
            if "val_acc" in print_args:
                max_ind = val_acc.index(max(val_acc))
                print("Best Validation Loss and Accuracy: ", val_losses[max_ind], np.array(val_acc).max())
                print("Test Loss and Accuracy: ", tst_losses[max_ind], tst_acc[max_ind])
            else:
                print("Best Validation Loss: ", val_losses[-1])

        if "tst_loss" in print_args:
            if "tst_acc" in print_args:
                max_ind = tst_acc.index(max(tst_acc))
                print("Best Test Data Loss and Accuracy: ", tst_losses[max_ind], np.array(tst_acc).max())
            else:
                print("Best Test Data Loss: ", tst_losses[-1])

        print('-----------------------------------')
        print(self.configdata['ds_strategy']['type'], file=logfile)
        print('---------------------------------------------------------------------', file=logfile)

        if "trn_acc" in print_args:
            trn_str = "Train Accuracy, "
            for trn in trn_acc:
                trn_str = trn_str + " , " + str(trn)
            print(trn_str, file=logfile)

        if "val_acc" in print_args:
            val_str = "Validation Accuracy, "
            for val in val_acc:
                val_str = val_str + " , " + str(val)
            print(val_str, file=logfile)

        if "tst_acc" in print_args:
            tst_str = "Test Accuracy, "
            for tst in tst_acc:
                tst_str = tst_str + " , " + str(tst)
            print(tst_str, file=logfile)
       
        logfile.close()


torch.autograd.set_detect_anomaly(True)

tc = TrainClassifier(args.con_file)

tc.train()
