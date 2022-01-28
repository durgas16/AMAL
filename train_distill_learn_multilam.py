import time
import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from torch.utils.data import Dataset
import os.path as osp

import torch.nn.functional as F

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

from torch.distributions import Categorical

from utils.custom_dataset import load_dataset_custom
from utils.config_utils import load_config_data
from utils.scheduler import Scheduler

from models import *
from models.resnet_cifar import resnet8_cifar, resnet20_cifar, resnet110_cifar

#from getLambda.Lambda import RewLambda
#from getLambda.LearnLambdaMeta import LearnLambdaMeta
from getLambda.LearnMultiLambdaMeta import LearnMultiLambdaMeta

seed = 84 # [42,36,24,16,50]

# os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.manual_seed(seed)
np.random.seed(seed)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

Temp = 4
_lambda = 0.1#.1

print(seed,_lambda)

class MyDataset(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]

        # Your transformations here (or set it in CIFAR10)

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
        elif mtype == 'WRN_16_X':
            print(mtype , d, w )
        else:
            print(mtype)

        if mtype == 'ResNet18':
            model = ResNet18(self.configdata['model']['numclasses'])
        elif mtype == 'ResNet50':
            model = ResNet50(self.configdata['model']['numclasses'])
        elif mtype == 'ResNet8':
            model = resnet8_cifar(num_classes=self.configdata['model']['numclasses'])
        elif mtype == 'ResNet14':
            model = resnet14_cifar(num_classes=self.configdata['model']['numclasses'])
        elif mtype == 'ResNet56':
            model = resnet56_cifar(num_classes=self.configdata['model']['numclasses'])
        elif mtype == 'ResNet26':
            model = resnet26_cifar(num_classes=self.configdata['model']['numclasses'])
        elif mtype == 'ResNet32':
            model = resnet32_cifar(num_classes=self.configdata['model']['numclasses'])
        elif mtype == 'WideResNet':
            model = wrn(input_shape=self.configdata['model']['input_shape'], \
                        num_classes=self.configdata['model']['numclasses'], \
                        depth=28, widen_factor=10, repeat=3, dropRate=0.3, bias=True)

        elif mtype == 'WRN_16_X':
            if self.configdata['dataset']['name'] in ['cars','flowers','airplane','dogs','Cub2011']:
                model = WRN_16_X(depth=d, width =w,num_classes=self.configdata['model']['numclasses'],if_large=True)
            else:
                model = WRN_16_X(depth=d, width =w,num_classes=self.configdata['model']['numclasses'],if_large=False)

        elif mtype == 'NN_2L':
            model = TwoLayerNet(input_dim=self.configdata['model']['input_dims'], \
                                num_classes=self.configdata['model']['numclasses'], \
                                hidden_units=hid_unit)

        elif mtype == 'Linear':
            model = LogisticRegNet(input_dim=self.configdata['model']['input_dims'], \
                                   num_classes=self.configdata['model']['numclasses'])

        '''elif mtype == 'resnext50_32x4d':
            model = resnext50_32x4d(num_classes=self.configdata['model']['numclasses'])'''

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
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80],
                                                             gamma=0.5)  # [60,120,160],
        elif self.configdata['scheduler']['type'] == 'Mstep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=\
            [int(elem*self.configdata['train_args']['num_epochs']) for elem in [0.3, 0.6, 0.8]], gamma=0.2)                                                     
        elif self.configdata['scheduler']['type'] == 'RPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', \
                                                                   factor=self.configdata['scheduler']['decay'],
                                                                   patience=10, threshold=0.0001, \
                                                                   threshold_mode='rel', cooldown=5, min_lr=1e-08,
                                                                   eps=1e-07, verbose=False)
        elif self.configdata['scheduler']['type'] == 'cyclic_cosine':
            scheduler = Scheduler(optimizer, self.configdata['scheduler']['lr'], self.configdata['scheduler']['lr_max'], \
                                  self.configdata['scheduler']['lr_max_decay'], self.configdata['scheduler']['lr_min'], \
                                  self.configdata['scheduler']['lr_min_decay'],
                                  self.configdata['ds_strategy']['schedule'])

            scheduler.sch_ind = self.configdata['ds_strategy']['sch_ind']
        return optimizer, scheduler

    def generate_cumulative_timing(self, mod_timing):
        tmp = 0
        mod_cum_timing = np.zeros(len(mod_timing))
        for i in range(len(mod_timing)):
            tmp += mod_timing[i]
            mod_cum_timing[i] = tmp
        return mod_cum_timing / 3600

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

        valid = True

        if valid:
            if self.configdata['dataset']['feature'] == 'classimb':
                trainset, validset, testset, num_cls = load_dataset_custom(self.configdata['dataset']['datadir'],
                                                                           self.configdata['dataset']['name'],
                                                                           self.configdata['dataset']['feature'],
                                                                           classimb_ratio=self.configdata['dataset'][
                                                                               'classimb_ratio'], valid=valid)
            else:
                if self.configdata['dataset']['name'] == "synthetic":
                    trainset,noise_lb, validset, testset, num_cls = load_dataset_custom(self.configdata['dataset']['datadir'],
                                                                               self.configdata['dataset']['name'],
                                                                               self.configdata['dataset']['feature'],
                                                                               seed=42,
                                                                               valid=valid,
                                                                               type=self.configdata['dataset']['type'], \
                                                                               test_type=self.configdata['dataset'][
                                                                                   'test_type'], \
                                                                               ncls=self.configdata['model'][
                                                                                   'numclasses'])  # noise_ratio=0.5) #24
                else:
                    trainset, validset, testset, num_cls = load_dataset_custom(self.configdata['dataset']['datadir'],
                                                                               self.configdata['dataset']['name'],
                                                                               self.configdata['dataset']['feature'],
                                                                               valid=valid)

                    '''trainsetN, validsetN, _,_ = load_dataset_custom(self.configdata['dataset']['datadir'],
                                                                               self.configdata['dataset']['name'],
                                                                               self.configdata['dataset']['feature'],
                                                                               valid=valid,transformN=True
                                                                    )'''

        else:
            if self.configdata['dataset']['feature'] == 'classimb':
                trainset, testset, num_cls = load_dataset_custom(self.configdata['dataset']['datadir'],
                                                                 self.configdata['dataset']['name'],
                                                                 self.configdata['dataset']['feature'],
                                                                 classimb_ratio=self.configdata['dataset'][
                                                                     'classimb_ratio'], valid=valid)
            else:
                if self.configdata['dataset']['name'] == "synthetic":
                    trainset,noise_lb, testset, num_cls = load_dataset_custom(self.configdata['dataset']['datadir'],
                                                                     self.configdata['dataset']['name'],
                                                                     self.configdata['dataset']['feature'], seed=42,
                                                                     valid=valid,
                                                                     type=self.configdata['dataset']['type'], \
                                                                     test_type=self.configdata['dataset']['test_type'], \
                                                                     ncls=self.configdata['model'][
                                                                         'numclasses'])  # ,noise_ratio=0.5) #24
                else:
                    trainset, testset, num_cls = load_dataset_custom(self.configdata['dataset']['datadir'],
                                                                     self.configdata['dataset']['name'],
                                                                     self.configdata['dataset']['feature'],
                                                                     valid=valid)

        # print(trainset.dataset.data.shape)
        N = len(trainset)
        trn_batch_size = self.configdata['dataloader']['trn_batch_size']
        val_batch_size = self.configdata['dataloader']['val_batch_size']
        tst_batch_size = self.configdata['dataloader']['tst_batch_size']

        # Creating the Data Loaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                                  shuffle=False, pin_memory=True)


        trainloaderS = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                                   shuffle=True, pin_memory=True)

        trainloader_ind = torch.utils.data.DataLoader(MyDataset(trainset), batch_size=trn_batch_size,
                                                  shuffle=True, pin_memory=True)

        trainloader_ind_small = torch.utils.data.DataLoader(MyDataset(trainset), batch_size=trn_batch_size//4,
                                                  shuffle=True, pin_memory=True)

        if valid:
            valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size,
                                                    shuffle=True, pin_memory=True)

            for batch_idx, (inputs, targets) in enumerate(valloader):
                if batch_idx == 0:
                    full_targets_val = targets
                else:
                    full_targets_val = torch.cat((full_targets_val, targets), dim=0)

            print(torch.unique(full_targets_val,return_counts=True))

        testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
                                                 shuffle=False, pin_memory=True)

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx == 0:
                full_targets_train = targets
            else:
                full_targets_train = torch.cat((full_targets_train, targets), dim=0)

        print(torch.unique(full_targets_train,return_counts=True))

        trn_losses = list()
        val_losses = list()  # np.zeros(configdata['train_args']['num_epochs'])
        tst_losses = list()
        subtrn_losses = list()
        timing = list()
        trn_acc = list()
        val_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])
        tst_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])
        subtrn_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])
        #batches = list()

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
            if self.configdata['model']['architecture'] == 'WRN_16_X':
                if _lambda > 0:
                    all_logs_dir = os.path.join(results_dir, self.configdata['ds_strategy']['type'] + "_distilT",\
                    self.configdata['dataset']['name'],self.configdata['model']['architecture']+"_"+\
                    str(self.configdata['model']['depth_teach'])+"_"+str(self.configdata['model']['width_teach']) +"_"+\
                    str(self.configdata['model']['depth'])+ "_"+str(self.configdata['model']['width'])+"_p" + \
                    str(_lambda * 10),str(seed))
                else:
                    all_logs_dir = os.path.join(results_dir, self.configdata['ds_strategy']['type'] + "_distilT",\
                    self.configdata['dataset']['name'],self.configdata['model']['architecture']+"_"+\
                    str(self.configdata['model']['depth'])+ "_"+str(self.configdata['model']['width'])+"_p" + \
                    str(_lambda * 10),str(seed))

            else:
                all_logs_dir = os.path.join(results_dir, self.configdata['ds_strategy']['type'] + "_distilT",
                                            self.configdata['dataset']['name'],
                                            self.configdata['model']['architecture'] + "_p" + str(_lambda * 10),str(seed))
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
            if self.configdata['model']['architecture'] == 'WRN_16_X':
                if _lambda > 0:
                    ckpt_dir = os.path.join(checkpoint_dir, self.configdata['ds_strategy']['type'] + "_distilT",\
                    self.configdata['dataset']['name'],self.configdata['model']['architecture']+"_"+\
                    str(self.configdata['model']['depth_teach'])+"_"+str(self.configdata['model']['width_teach']) +"_"+\
                    str(self.configdata['model']['depth'])+ "_"+str(self.configdata['model']['width'])+"_p" + \
                    str(_lambda * 10),str(seed))
                else:
                    ckpt_dir = os.path.join(checkpoint_dir, self.configdata['ds_strategy']['type'] + "_distilT",\
                    self.configdata['dataset']['name'],self.configdata['model']['architecture']+"_"+\
                    str(self.configdata['model']['depth'])+ "_"+str(self.configdata['model']['width'])+"_p" + \
                    str(_lambda * 10),str(seed))

            else:
                ckpt_dir = os.path.join(checkpoint_dir, self.configdata['ds_strategy']['type'] + "_distilT",
                                        self.configdata['dataset']['name'],
                                        self.configdata['model']['architecture'] + "_p" + str(_lambda * 10),str(seed))

        checkpoint_path = os.path.join(ckpt_dir, 'model.pt')
        checkpoint_path_10 = os.path.join(ckpt_dir, 'model_10.pt')
        os.makedirs(ckpt_dir, exist_ok=True)

        torch.manual_seed(seed)

        # Model Creation
        
        mtype = self.configdata['model']['architecture']
        hid_unit=None
        d=None
        w=None
        if mtype == 'NN_2L':
            hid_unit = self.configdata['model']['hidden_units_stu']
        elif mtype == 'WRN_16_X':
            d = self.configdata['model']['depth']
            w = self.configdata['model']['width']
        train_model = self.create_model(mtype,hid_unit,d,w)
        print("Student",sum(p.numel() for p in train_model.parameters() if p.requires_grad))
        #ema_model = torch.nn.DataParallel(self.create_model(), device_ids=[0, 1])
        
        Nteacher = 1
        if _lambda > 0:
            mtype = self.configdata['model']['teacher_arch']
            hid_unit=None
            d=None
            w=None
            if 'NN_2L' in mtype:
                hid_unit = self.configdata['model']['hidden_units_teach']
            elif 'WRN_16_X' in mtype:
                d = self.configdata['model']['depth_teach']
                w = self.configdata['model']['width_teach']
            
            teacher_model = []
            Nteacher = len(mtype)
            
            for m in range(len(mtype)):

                if mtype[m] == 'NN_2L':
                   teacher_model.append(self.create_model(mtype[m],hid_unit=hid_unit[m]))
                elif mtype[m] == 'WRN_16_X':
                    teacher_model.append(self.create_model(mtype[m],d=d[m],w=w[m]))
                else:
                    teacher_model.append(self.create_model(mtype[m]))
                
                print("Teacher",sum(p.numel() for p in teacher_model[-1].parameters() if p.requires_grad))
                print("Loading from",self.configdata['model']['teacher_path'][m])
                checkpoint = torch.load(self.configdata['model']['teacher_path'][m])
                teacher_model[-1].load_state_dict(checkpoint['state_dict'])
                #teacher_model.append(copy.deepcopy(model))

        # Loss Functions
        criterion, criterion_nored = self.loss_function()

        # Getting the optimizer and scheduler
        optimizer, scheduler = self.optimizer_with_scheduler(train_model)
        
        if self.configdata['ds_strategy']['type'] in ['MultiLam']:
            lambdas = torch.rand((N,Nteacher+1), device=self.configdata['train_args']['device'])
            
            
            lambdas[:,0] = 1- torch.max(lambdas[:,1:],dim=1).values
            
            for m in range(Nteacher+1):
                print(lambdas[:,m].max(), lambdas[:,m].min(), torch.median(lambdas[:,m]),\
                        torch.quantile(lambdas[:,m], 0.75),torch.quantile(lambdas[:,m], 0.25))
        all_outputs =[]
        print("=======================================", file=logfile)

        if self.configdata['ckpt']['is_load'] == True:
            if self.configdata['ds_strategy']['type'] in ['MultiLam']:
                start_epoch, train_model, optimizer, scheduler, ckpt_loss, load_metrics, all_lambda,all_outputs= self.load_ckp(
                    checkpoint_path_10, train_model, optimizer, scheduler) #ema_model,
                lambdas = copy.deepcopy(all_lambda[-1])
            else:
                start_epoch, train_model, optimizer, scheduler, ckpt_loss, load_metrics, all_outputs= self.load_ckp(
                    checkpoint_path, train_model, optimizer, scheduler) 

            print("Loading saved checkpoint model at epoch " + str(start_epoch))
            
            #soft_lam = F.softmax(lambdas, dim=1)
            for arg in load_metrics.keys():
                if valid:
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
                if arg == "time":
                    timing = load_metrics['time']

            sch_ind = 0
            while start_epoch > self.configdata['ds_strategy']['schedule'][sch_ind]:
                sch_ind += 1

            if self.configdata['ds_strategy']['type'] not in ['No-curr']:
                select = self.configdata['ds_strategy']['select_every']
        else:
            start_epoch = 0

        decay = self.configdata['ds_strategy']['decay']

        '''if self.configdata['ds_strategy']['type'] in ['RewLam']:

            rel = RewLambda(trainloader_ind, valloader, train_model, num_cls, N, criterion_nored, \
                            self.configdata['train_args']['device'], self.configdata['dataset']['grad_fit'], \
                            teacher_model, criterion, Temp)

        elif self.configdata['ds_strategy']['type'] in ['LearnLam']:

            lelam = LearnLambdaMeta(trainloader_ind, valloader, train_model, num_cls, N, criterion_nored, \
                            self.configdata['train_args']['device'], self.configdata['dataset']['grad_fit'], \
                            teacher_model, criterion, Temp) #testloader'''
        
        if self.configdata['ds_strategy']['type'] in ['MultiLam']:
            lelam = LearnMultiLambdaMeta(trainloader_ind_small, valloader, train_model, num_cls, N, criterion_nored, \
                                self.configdata['train_args']['device'], self.configdata['dataset']['grad_fit'], \
                                teacher_model, criterion, Temp)
            
        for i in range(start_epoch, self.configdata['train_args']['num_epochs']):

            subtrn_loss = 0
            subtrn_correct = 0
            subtrn_total = 0

            train_model.train()

            if i in self.configdata['ds_strategy']['schedule'][1:] and i >= self.configdata['ds_strategy'][
                'warm_epoch']:
                ind = self.configdata['ds_strategy']['schedule'].index(i)

            if self.configdata['ds_strategy']['type'] in ['MultiLam']:

                select = self.configdata['ds_strategy']['select_every']

            if (i < self.configdata['ds_strategy']['warm_epoch']) or \
                    (self.configdata['ds_strategy']['type'] in ['No-curr']):  # or\

                start_time = time.time()
               
                for batch_idx, (inputs, targets,indices) in enumerate(trainloader_ind):
                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                        self.configdata['train_args']['device'],
                        non_blocking=True)  # targets can have non_blocking=True.
                    optimizer.zero_grad()
                    outputs = train_model(inputs)

                    if _lambda > 0:                            
                        # print(torch.sum(torch.isnan(F.log_softmax(outputs / Temp, dim=1))),torch.sum(torch.isnan(F.softmax(teacher_outputs / Temp, dim=1))),loss_KD)
                        if self.configdata['ds_strategy']['type'] in ['MultiLam']:
                            loss_SL = criterion_nored(outputs, targets)
                            loss = lambdas[indices,0]*loss_SL
                            for m in range(Nteacher):
                                with torch.no_grad():
                                    teacher_outputs = teacher_model[m](inputs)
                                loss_KD = nn.KLDivLoss(reduction='none')(
                                    F.log_softmax(outputs / Temp, dim=1), F.softmax(teacher_outputs / Temp, dim=1))
                                loss_KD =  Temp * Temp * torch.sum(loss_KD, dim=1)
                                loss += lambdas[indices,m+1]*loss_KD
                            #print(loss_SL,loss_KD)
                            loss = torch.mean(loss)
                        else:
                            loss_SL = criterion(outputs, targets)
                            loss = (1 - _lambda) * loss_SL
                            for m in range(Nteacher):
                                with torch.no_grad():
                                    teacher_outputs = teacher_model[m](inputs)
                                #print(teacher_outputs )
                                loss_KD = nn.KLDivLoss(reduction='batchmean')(
                                    F.log_softmax(outputs / Temp, dim=1), F.softmax(teacher_outputs / Temp, dim=1))
                                loss += _lambda * Temp * Temp * loss_KD

                    else:
                        loss = criterion(outputs, targets)

                    loss.backward()
                    subtrn_loss += loss.item()
                    optimizer.step()
                    del inputs, targets

                print("Training with lr", round(optimizer.param_groups[0]['lr'], 5))
                if self.configdata['scheduler']['type'] == 'cyclic_cosine':
                    scheduler.adjust_cosine_learning_rate_step(i + 1)
                elif self.configdata['scheduler']['type'] == 'RPlateau':
                    if valid and len(val_losses) > 0:
                        scheduler.step(val_losses[-1])
                    elif len(trn_losses) > 0:
                        scheduler.step(trn_losses[-1])
                else:
                    scheduler.step()

                train_time = time.time() - start_time

            elif self.configdata['ds_strategy']['type'] in ['MultiLam']:

                '''if i in self.configdata['ds_strategy']['schedule']:
                    ind = self.configdata['ds_strategy']['schedule'].index(i)'''

                select = self.configdata['ds_strategy']['select_every']

                if (i % select == 0 or i in self.configdata['ds_strategy']['schedule']):

                    cached_state_dictT = copy.deepcopy(train_model.state_dict())
                    #cached_state_dict = copy.deepcopy(ema_model.state_dict())
                    clone_dict = copy.deepcopy(train_model.state_dict())
                    #if self.configdata['ds_strategy']['type'] in ['MultiLam']:
                    lelam.update_model(clone_dict)
                    #ema_model.load_state_dict(cached_state_dict)
                    train_model.load_state_dict(cached_state_dictT)

                    if self.configdata['ds_strategy']['type'] in ['MultiLam']:
                        lambdas = lelam.get_lambdas(optimizer.param_groups[0]['lr'],i,lambdas)
                        #soft_lam = F.softmax(lambdas, dim=1)
                        for m in range(Nteacher+1):
                            #print(soft_lam[:,m].max(), soft_lam[:,m].min(), torch.median(soft_lam[:,m]),\
                            #     torch.quantile(soft_lam[:,m], 0.75),torch.quantile(soft_lam[:,m], 0.25))
                            print(lambdas[:,m].max(), lambdas[:,m].min(), torch.median(lambdas[:,m]),\
                                 torch.quantile(lambdas[:,m], 0.75),torch.quantile(lambdas[:,m], 0.25))
                   

                print("Training with lr", round(optimizer.param_groups[0]['lr'], 5))
                start_time = time.time()

                for batch_idx, (inputs, targets,indices) in enumerate(trainloader_ind):


                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                        self.configdata['train_args']['device'],
                        non_blocking=True)  # targets can have non_blocking=True.
                    optimizer.zero_grad()

                    outputs = train_model(inputs)

                    loss_SL = criterion_nored(outputs, targets)
                    #loss = soft_lam[indices,0]*loss_SL
                    loss = lambdas[indices,0]*loss_SL
                    for m in range(Nteacher):
                        with torch.no_grad():
                            teacher_outputs = teacher_model[m](inputs)
                        loss_KD = nn.KLDivLoss(reduction='none')(
                            F.log_softmax(outputs / Temp, dim=1), F.softmax(teacher_outputs / Temp, dim=1))
                        loss_KD =  Temp * Temp * torch.sum(loss_KD, dim=1)
                        #loss += soft_lam[indices,m+1]*loss_KD
                        loss += lambdas[indices,m+1]*loss_KD
                    loss = torch.mean(loss)
                    loss.backward()
                    subtrn_loss += loss.item()
                    optimizer.step()

                # update_ema_variables(train_model, ema_model, decay,i)#-self.configdata['ds_strategy']['schedule'][sc])
                if self.configdata['scheduler']['type'] == 'cyclic_cosine':
                    scheduler.adjust_cosine_learning_rate_step(i + 1)
                elif self.configdata['scheduler']['type'] == 'RPlateau':
                    if valid:
                        scheduler.step(val_losses[-1])
                    else:
                        scheduler.step(trn_losses[-1])
                else:
                    # if (i+1)%5 == 0:
                    scheduler.step()
                train_time = time.time() - start_time

            timing.append(train_time)  # + subset_selection_time)
            print_args = self.configdata['train_args']['print_args']
            # print("Epoch timing is: " + str(timing[-1]))

            trn_loss = 0
            trn_correct = 0
            trn_total = 0
            if valid:
                val_loss = 0
                val_correct = 0
                val_total = 0
            tst_correct = 0
            tst_total = 0
            tst_loss = 0
            train_model.eval()

            # if "trn_loss" in print_args:
            if _lambda > 0 and i<=1:
                diff_teach = torch.zeros(N).bool()
            entropy_trn = torch.zeros(N)
            target_trn = torch.zeros(N).int()
            curr_outputs = torch.zeros(N,num_cls)
            with torch.no_grad():
                for batch_idx, (inputs, targets,indices) in enumerate(trainloader_ind):
                    # print(batch_idx)
                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                        self.configdata['train_args']['device'], non_blocking=True)
                    outputs = train_model(inputs)
                    curr_outputs[indices] = outputs.cpu()
                    if _lambda > 0 and i<=1:
                        teacher_outputs = teacher_model[0](inputs)
                        _, tpredicted = teacher_outputs.max(1)
                        
                        diff_teach[indices] = tpredicted.eq(targets).cpu()

                    entropy_trn[indices] = Categorical(probs=F.softmax(outputs, dim=1)).entropy().cpu()

                    loss = criterion(outputs, targets)
                    trn_loss += loss.item()
                    # if "trn_acc" in print_args:
                    _, predicted = outputs.max(1)
                    trn_total += targets.size(0)
                    trn_correct += predicted.eq(targets).sum().item()
                trn_losses.append(trn_loss)
                trn_acc.append(trn_correct / trn_total)
                #print(diff_teach,trn_acc[-1])
            all_outputs.append(curr_outputs)
            
            '''if (i < self.configdata['ds_strategy']['warm_epoch']) or \
                    (self.configdata['ds_strategy']['type'] in ['No-curr']) or \
                    ((i + 1) % select == 0 or (i + 1) in self.configdata['ds_strategy']['schedule']):

                ema_model.load_state_dict(copy.deepcopy(train_model.state_dict()))
            else:
                update_ema_variables(train_model, ema_model, 0.2, i - 1)'''

            if "val_loss" in print_args and valid:
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(valloader):
                        # print(batch_idx)
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                            self.configdata['train_args']['device'], non_blocking=True)
                        outputs = train_model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()

                        if "val_acc" in print_args:
                            _, predicted = outputs.max(1)
                            val_total += targets.size(0)
                            val_correct += predicted.eq(targets).sum().item()
                    val_losses.append(val_loss)
                    val_acc.append(val_correct / val_total)

            if "tst_loss" in print_args:
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(testloader):
                        # print(batch_idx)
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                            self.configdata['train_args']['device'], non_blocking=True)
                        outputs = train_model(inputs)
                        loss = criterion(outputs, targets)
                        tst_loss += loss.item()

                        if "tst_acc" in print_args:
                            _, predicted = outputs.max(1)
                            tst_total += targets.size(0)
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

                    tst_losses.append(tst_loss)
                    tst_acc.append(tst_correct / tst_total)
                    if i + 1 == self.configdata['train_args']['num_epochs']:
                        macro = precision_recall_fscore_support(full_targets.cpu().numpy(), full_predict.cpu().numpy(),
                                                                average='macro')
                        micro = precision_recall_fscore_support(full_targets.cpu().numpy(), full_predict.cpu().numpy(),
                                                                average='micro')

                        matrix = confusion_matrix(full_targets.cpu().numpy(), full_predict.cpu())
                        np.save(osp.join(all_logs_dir, 'confusion.npy'), matrix)
                        print(matrix)
                        tloss = Categorical(probs=F.softmax(full_logit, dim=1)).entropy()
                        print(tloss.max(), tloss.min(), torch.median(tloss), torch.quantile(tloss, 0.75),
                              torch.quantile(tloss, 0.25))
            '''if (i < self.configdata['ds_strategy']['warm_epoch']) or \
                    (self.configdata['ds_strategy']['type'] in ['No-curr']) or \
                    self.configdata['ds_strategy']['type'] in ['Reweigh', 'RewLam']:
                batches.append(N // trn_batch_size + 1)
            else:
                batches.append(len(idxs) // trn_batch_size + 1)'''

            if self.configdata['ds_strategy']['type'] in ['MultiLam'] and (i % select == 0 or i in self.configdata['ds_strategy']['schedule']):
                if (i < self.configdata['ds_strategy']['warm_epoch']):
                    all_lambda = []
                
                all_lambda.append(copy.deepcopy(lambdas))

            if ((i + 1) % self.configdata['train_args']['print_every'] == 0):

                if "subtrn_acc" in print_args:
                    subtrn_acc.append(subtrn_correct / subtrn_total)

                if "subtrn_losses" in print_args:
                    subtrn_losses.append(subtrn_loss)

                print_str = "Epoch: " + str(i + 1)

                for arg in print_args:

                    if valid:

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

                    if arg == "subtrn_loss":
                        print_str += " , " + "Subset Loss: " + str(subtrn_losses[-1])

                    if arg == "subtrn_acc":
                        print_str += " , " + "Subset Accuracy: " + str(subtrn_acc[-1])

                    if arg == "time":
                        print_str += " , " + "Timing: " + str(timing[-1])

                # report metric to ray for hyperparameter optimization
                # if 'report_tune' in self.configdata and self.configdata['report_tune']:
                #    tune.report(mean_accuracy=val_acc[-1])

                print(print_str)

            if ((i + 1) % self.configdata['ckpt']['save_every'] == 0):

                cmap = plt.cm.jet
                # extract all colors from the .jet map
                cmaplist = [cmap(i) for i in range(cmap.N)]
                # create the new map
                cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

                # define the bins and normalize
                bounds = np.linspace(0, num_cls, num_cls + 1)
                norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

                if self.configdata['ckpt']['is_save'] == True:

                    metric_dict = {}

                    for arg in print_args:

                        if valid:
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
                        if arg == "subtrn_loss":
                            metric_dict['subtrn_loss'] = subtrn_losses
                        if arg == "subtrn_acc":
                            metric_dict['subtrn_acc'] = subtrn_acc
                        if arg == "time":
                            metric_dict['time'] = timing

                    #metric_dict['batches'] = batches

                    ckpt_state = {
                        'epoch': i + 1,
                        'state_dict': train_model.state_dict(),
                        #'state_dict_ema': ema_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': self.loss_function(),
                        'metrics': metric_dict,
                        'output': all_outputs
                        
                    }

                    if self.configdata['ds_strategy']['type'] in ['MultiLam']:
                        flambda = all_lambda[-1][:,1].cpu()

                        print(flambda.max(), flambda.min(), torch.median(flambda), torch.quantile(flambda, 0.75),torch.quantile(flambda, 0.25))
                        ckpt_state['lambda']= all_lambda

                    if self.configdata['dataset']['name'] == "synthetic":
                        ckpt_state['noise_lb']= noise_lb

                    # save checkpoint
                    self.save_ckpt(ckpt_state, checkpoint_path)
                    if (i + 1) == 10:
                        self.save_ckpt(ckpt_state, checkpoint_path_10)
                    print("Model checkpoint saved at epoch " + str(i + 1))

                elif self.configdata['ckpt']['is_save_pic'] == True:

                    if ((trainset.dataset[trainset.indices])[0]).shape[1] == 2:
                        plt.figure()
                        plt.title("Training @ " + str(i + 1) + " epoch")

                        '''y = copy.deepcopy(trainset.targets)
                        if i+1 > 6:
                            y[idxs] = 10'''

                        if  ((i + 1) / self.configdata['ckpt']['save_every'] == 1):
                            my_scatter_plot = plt.scatter((trainset.dataset[trainset.indices])[0][:, 0],
                                                          (trainset.dataset[trainset.indices])[0][:, 1],
                                                          c=(trainset.dataset[trainset.indices])[1],
                                                          s=10, cmap=cmap, norm=norm)

                            '''if i + 1 > 11 and self.configdata['ds_strategy']['type'] not in ['No-curr','RewLam']:
                                my_scatter_plot = plt.scatter((trainset.dataset[trainset.indices])[0][idxs][:, 0],
                                                              (trainset.dataset[trainset.indices])[0][idxs][:, 1],
                                                              c='pink',
                                                              vmin=min((trainset.dataset[trainset.indices])[1]),
                                                              vmax=max((trainset.dataset[trainset.indices])[1]),
                                                              s=10)'''
                            plt.savefig(os.path.join(all_logs_dir, 'train_' + str(i + 1) + '.png'))
                            #plt.show()
                            plt.close()

                            
                            my_scatter_plot = plt.scatter((trainset.dataset[trainset.indices])[0][~diff_teach][:, 0],
                                                          (trainset.dataset[trainset.indices])[0][~diff_teach][:, 1],
                                                          c=(trainset.dataset[trainset.indices])[1][~diff_teach],
                                                          s=10, cmap=cmap, norm=norm)

                            '''if i + 1 > 11 and self.configdata['ds_strategy']['type'] not in ['No-curr','RewLam']:
                                my_scatter_plot = plt.scatter((trainset.dataset[trainset.indices])[0][idxs][:, 0],
                                                              (trainset.dataset[trainset.indices])[0][idxs][:, 1],
                                                              c='pink',
                                                              vmin=min((trainset.dataset[trainset.indices])[1]),
                                                              vmax=max((trainset.dataset[trainset.indices])[1]),
                                                              s=10)'''
                            plt.savefig(os.path.join(all_logs_dir, 'train_diff' + str(i + 1) + '.png'))
                            #plt.show()
                            plt.close()

                        if i + 1 > 11 and self.configdata['ds_strategy']['type'] in ['MultiLam']:
                            #fig, ax = plt.subplots()

                            plt.figure()
                            """plt.title("Lambdas @ " + str(i + 1) + " epoch")

                            curr_lam = all_lambda[-1]
                            g_lam = torch.zeros_like(curr_lam)
                            for qu in range(1,10):
                                g_lam[curr_lam > qu/10] = qu/10

                            g_lam_list = [str(round(qu,2))+"-"+str(round(qu+.1,2)) for qu in g_lam.tolist()]
                            scatter = plt.scatter((trainset.dataset[trainset.indices])[0][:, 0],
                                        (trainset.dataset[trainset.indices])[0][:, 1],
                                        c=g_lam.cpu().numpy(),s=10) #vmin=min(g_lam_list),vmax=max(g_lam_list),

                            '''legend1 = plt.legend(*scatter.legend_elements(),
                                                loc="lower left", title="Percentile")
                            plt.add_artist(legend1)'''

                            had, lab = scatter.legend_elements()
                            plt.legend(had,g_lam_list,loc='upper left',bbox_to_anchor=(1,0.5),title="Lambdas")"""

                            curr_lam = all_lambda[-1][:,1]
                            g_lam = torch.zeros_like(curr_lam)
                            for qu in range(0, 10,3):
                                g_lam[curr_lam > qu / 10] = qu / 10

                            df = pd.DataFrame()
                            df["y"] = (trainset.dataset[trainset.indices])[1]
                            df["comp-1"] = (trainset.dataset[trainset.indices])[0][:, 0]
                            df["comp-2"] = (trainset.dataset[trainset.indices])[0][:, 1]
                            df["g_lam"] = [str(round(qu, 2)) + "-" + str(round(qu + .3, 2)) for qu in g_lam.tolist()]
                            df.sort_values(by=['g_lam'])


                            sns.scatterplot(x="comp-1", y="comp-2", data=df, hue="g_lam", \
                                            palette=sns.color_palette("Paired", len(df["g_lam"].unique()))).set(
                                title="Lambdas @ " + str(i + 1) + " epoch")
                            plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5),title="Lambdas")
                            plt.savefig(os.path.join(all_logs_dir, 'lam_' + str(i + 1) + '.png'),bbox_inches='tight')
                            #plt.show()
                            plt.close()


                            dft = pd.DataFrame()
                            dft["y"] = (trainset.dataset[trainset.indices])[1][~diff_teach]
                            dft["comp-1"] = (trainset.dataset[trainset.indices])[0][~diff_teach][:, 0]
                            dft["comp-2"] = (trainset.dataset[trainset.indices])[0][~diff_teach][:, 1]
                            dft["g_lam"] = [str(round(qu, 2)) + "-" + str(round(qu + .1, 2)) for qu in g_lam[~diff_teach].tolist()]
                            dft.sort_values(by=['g_lam'])
                            sns.scatterplot(x="comp-1", y="comp-2", data=dft, hue="g_lam", \
                                            palette=sns.color_palette("Paired", len(dft["g_lam"].unique()))).set(
                                title="Lambdas @ " + str(i + 1) + " epoch for different lables")
                            plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5),title="Lambdas")
                            plt.savefig(os.path.join(all_logs_dir, 'lam_diff' + str(i + 1) + '.png'),bbox_inches='tight')
                            #plt.show()
                            plt.close()

                            dft = pd.DataFrame()
                            dft["y"] = (trainset.dataset[trainset.indices])[1][noise_lb]
                            dft["comp-1"] = (trainset.dataset[trainset.indices])[0][noise_lb][:, 0]
                            dft["comp-2"] = (trainset.dataset[trainset.indices])[0][noise_lb][:, 1]
                            dft["g_lam"] = [str(round(qu, 2)) + "-" + str(round(qu + .1, 2)) for qu in g_lam[noise_lb].tolist()]
                            dft.sort_values(by=['g_lam'])
                            sns.scatterplot(x="comp-1", y="comp-2", data=dft, hue="g_lam", \
                                            palette=sns.color_palette("Paired", len(dft["g_lam"].unique()))).set(
                                title="Lambdas @ " + str(i + 1) + " epoch for different lables")
                            plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5),title="Lambdas")
                            plt.savefig(os.path.join(all_logs_dir, 'lam_noise' + str(i + 1) + '.png'),bbox_inches='tight')
                            #plt.show()
                            plt.close()

                            """#fig, ax = plt.subplots()
                            plt.figure()
                            entropy_trn = entropy_trn.cpu().numpy()
                            g_lam = np.zeros_like(entropy_trn)#*entropy_trn.min()
                            ent = np.linspace(0.0, np.log(num_cls)/np.log(2),10, endpoint=True)
                            ent = [round(qu,2) for qu in ent.tolist()]
                            for qu in range(1,10):
                                g_lam[entropy_trn > ent[qu]] = ent[qu]

                            g_lam = [round(qu,2) for qu in g_lam.tolist()]
                            #print(ent,g_lam[0])
                            #print(ent.index(g_lam[0]))#np.where(ent == g_lam[0]))
                            df["g_lam"] = [str(qu) + "-" + str(ent[ent.index(qu)+1]) for qu in g_lam]
                            sns.scatterplot(x="comp-1", y="comp-2", data=df, hue="g_lam", \
                                            palette=sns.color_palette("hls", len(df["g_lam"].unique()))).set(
                                title="Confidence @ " + str(i + 1) + " epoch")
                            plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5), title="Entropy")
                            plt.savefig(os.path.join(all_logs_dir, 'conf_' + str(i + 1) + '.png'), bbox_inches='tight')
                            #plt.show()
                            plt.close()"""

                    else:

                        plt.figure()
                        plt.title("Training @ " + str(i + 1) + " epoch")

                        X_embedded = TSNE(n_components=2).fit_transform((trainset.dataset[trainset.indices])[0])
                        df = pd.DataFrame()
                        df["y"] = (trainset.dataset[trainset.indices])[1]
                        df["comp-1"] = X_embedded[:, 0]
                        df["comp-2"] = X_embedded[:, 1]
                        sns.scatterplot(x="comp-1", y="comp-2", data=df, hue="y", \
                                        palette=sns.color_palette("hls", 20)).set(title="T-SNE projection")

                        '''if i + 1 > 6 and self.configdata['ds_strategy']['type'] not in ['No-curr','RewLam']:
                            df = pd.DataFrame()
                            df["y"] = (trainset.dataset[trainset.indices])[1][idxs]
                            df["comp-1"] = X_embedded[idxs][:, 0]
                            df["comp-2"] = X_embedded[idxs][:, 1]
                            sns.scatterplot(x="comp-1", y="comp-2", data=df, \
                                            hue="y", palette="pink").set(title="T-SNE projection")'''

                        plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
                        plt.savefig(os.path.join(all_logs_dir, 'train_' + str(i + 1) + '.png'),bbox_inches='tight')
                        plt.show()
                        plt.close()

                        if i + 1 > 6 and self.configdata['ds_strategy']['type'] in ['RewLam']:
                            plt.figure()
                            plt.title("Lambdas @ " + str(i + 1) + " epoch")

                            curr_lam = all_lambda[-1]
                            g_lam = torch.zeros_like(curr_lam)
                            for qu in range(1,10):
                                #lower = torch.quantile(lambdas, qu/10)
                                #upper = torch.quantile(lambdas, (qu+1)/10)
                                g_lam[curr_lam > qu/10] = qu/10

                            df["g_lam"] = [str(round(qu,2))+"-"+str(round(qu+.1,2)) for qu in g_lam.tolist()]
                            sns.scatterplot(x="comp-1", y="comp-2", data=df, hue="g_lam", \
                                            palette=sns.color_palette("hls", len(df["g_lam"].unique()))).set(title="Lambdas @ " + str(i + 1) + " epoch")
                            plt.legend(loc='upper left',bbox_to_anchor=(1,0.5))
                            plt.savefig(os.path.join(all_logs_dir, 'lam_' + str(i + 1) + '.png'),bbox_inches='tight')
                            plt.show()
                            plt.close()

                        """plt.figure()
                        plt.title("Test @ " + str(i + 1) + " epoch")

                        corr_idxs = full_predict.cpu().eq(testset.targets)

                        X_tst_embedded = TSNE(n_components=2).fit_transform(testset.data)
                        df = pd.DataFrame()
                        df["y"] = testset.targets[~corr_idxs]
                        df["comp-1"] = X_tst_embedded[~corr_idxs][:, 0]
                        df["comp-2"] = X_tst_embedded[~corr_idxs][:, 1]
                        sns.scatterplot(x="comp-1", y="comp-2", data=df,
                                        hue=df.y.tolist(),
                                        palette=sns.color_palette("hls", len(df["y"].unique()))).set(
                            title="T-SNE projection")

                        # print(corr_idxs)

                        '''my_scatter_plot = plt.scatter(testset.data[corr_idxs][:,0],
                                    testset.data[corr_idxs][:,1],
                                    c='r',
                                    vmin=min(testset.targets),
                                    vmax=max(testset.targets),
                                    s=10)'''
                        plt.text(0.2, 0.9, 'Test accuracy ' + str(tst_acc[-1] * 100), fontsize=10,
                                 transform=plt.gca().transAxes)
                        plt.savefig(os.path.join(all_logs_dir, 'test_' + str(i + 1) + '.png'))
                        plt.show()
                        plt.close()"""

        # np.save(osp.join(all_logs_dir,'entropy.npy'),np.stack(info))

        print(self.configdata['ds_strategy']['type'] + " Selection Run---------------------------------")
        print("Final SubsetTrn:", subtrn_loss)
        if "val_loss" in print_args and valid:
            if "val_acc" in print_args:
                print("Validation Loss and Accuracy: ", val_loss, np.array(val_acc).max())
            else:
                print("Validation Loss: ", val_loss)

        if "tst_loss" in print_args:
            if "tst_acc" in print_args:
                print("Test Data Loss and Accuracy: ", tst_loss, np.array(tst_acc).max())
            else:
                print("Test Data Loss: ", tst_loss)

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

        if "time" in print_args:
            time_str = "Time, "
            for t in timing:
                time_str = time_str + " , " + str(t)
            print(timing, file=logfile)

        print("Macro ", macro, file=logfile)
        print("Micro ", micro, file=logfile)
        omp_timing = np.array(timing)
        omp_cum_timing = list(self.generate_cumulative_timing(omp_timing))
        print("Total time taken by " + self.configdata['ds_strategy']['type'] + " = " + str(omp_cum_timing[-1]))
        logfile.close()


torch.autograd.set_detect_anomaly(True)

#tc = TrainClassifier("config/cifar100_wrn/config_no_curr_cifar100.py")
#tc = TrainClassifier("config/cifar100_wrn/config_multilam_cifar100.py")
#tc = TrainClassifier("config/cifar100_wrn/config_samemultilam_cifar_100.py")
#tc = TrainClassifier("config/cifar100_wrn/config_diffmultilam_cifar_100.py")
#tc = TrainClassifier("config/cifar100_wrn/config_no_curr_res_cifar100.py")

#tc = TrainClassifier("config/flowers/config_no_curr_flowers.py")
#tc = TrainClassifier("config/flowers/config_multilam_flowers.py")

#tc = TrainClassifier("config/cifar10_wrn/config_no_curr_cifar_10.py")
#tc = TrainClassifier("config/cifar10_wrn/config_learnlam_cifar_10.py")
#tc = TrainClassifier("config/cifar10_wrn/config_multilam_cifar_10.py")
#tc = TrainClassifier("config/cifar10_wrn/config_samemultilam_cifar_10.py")
#tc = TrainClassifier("config/cifar10_wrn/config_no_curr_multi_cifar_10.py")

tc = TrainClassifier("config/synthetic/config_learnlam_syn_gmm.py")
#tc = TrainClassifier("config/synthetic/config_no_curr_syn_gmm.py")

#tc = TrainClassifier("config/cars_wrn/config_no_curr_cars.py")
#tc = TrainClassifier("config/cars_wrn/config_multilam_cars.py")

#tc = TrainClassifier("config/dogs_wrn/config_no_curr_airplane.py")
#tc = TrainClassifier("config/dogs_wrn/config_multilam_airplane.py")
#tc = TrainClassifier("config/dogs_wrn/config_multilam_cub.py")
#tc = TrainClassifier("config/dogs_wrn/config_multilam_dogs.py")

#tc = TrainClassifier("config/dogs_wrn/config_no_curr_dogs.py")
#tc = TrainClassifier("config/dogs_wrn/config_no_curr_airplane.py")
#tc = TrainClassifier("config/dogs_wrn/config_no_curr_cub.py")

tc.train()

"""            elif self.configdata['ds_strategy']['type'] in ['LearnLam']:

                for batch_idx, (inputs, targets,indices) in enumerate(trainloader_ind):
                    
                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                        self.configdata['train_args']['device'],non_blocking=True)  # targets can have non_blocking=True.
                    lam_optimizer.zero_grad()
                    outputs = train_model(inputs)
                    teacher_outputs = teacher_model(inputs)
                    loss_SL = criterion_nored(outputs, targets)

                    loss_KD = nn.KLDivLoss(reduction='none')(F.log_softmax(outputs / Temp, dim=1), \
                                                                F.softmax(teacher_outputs / Temp, dim=1))
                    loss_KD = torch.sum(loss_KD, dim=1)

                    loss = torch.mean((1 - lambdas[indices]) * loss_SL + lambdas[indices] * Temp * Temp * loss_KD)

                    loss.backward(retain_graph=True)
                    subtrn_loss += loss.item()
                    optimizer.step()

                    optimizer.zero_grad()
                    iterator = iter(valloader)
                    data, label = next(iterator)

                    data, label = data.to(self.configdata['train_args']['device']), label.to(
                        self.configdata['train_args']['device'],non_blocking=True)  # targets can have non_blocking=True.
                    outputs = train_model(data)
                    val_loss = criterion(outputs, label)
                    val_loss.backward()
                    print(lambdas[indices[0]].grad)
                    lam_optimizer.step()

                    del inputs, targets

                print("Training with lr", round(optimizer.param_groups[0]['lr'], 5))
                if self.configdata['scheduler']['type'] == 'cyclic_cosine':
                    scheduler.adjust_cosine_learning_rate_step(i + 1)
                elif self.configdata['scheduler']['type'] == 'RPlateau':
                    if valid and len(val_losses) > 0:
                        scheduler.step(val_losses[-1])
                    elif len(trn_losses) > 0:
                        scheduler.step(trn_losses[-1])
                else:
                    scheduler.step()
                    lam_scheduler.step()

                train_time = time.time() - start_time"""

