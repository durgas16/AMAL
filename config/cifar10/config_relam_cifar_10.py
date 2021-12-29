# Learning setting
config = dict(setting="supervisedlearning",

              dataset=dict(name="cifar10",
                           datadir="../data",
                           feature="dss",
                           type="pre-defined",
                           grad_fit=3),

              dataloader=dict(shuffle=True,
                              trn_batch_size=128,
                              val_batch_size=1000,
                              tst_batch_size=1000,
                              pin_memory=True),

              model=dict(architecture='ResNet18',
                         type='pre-defined',
                         numclasses=10,
                         teacher_arch='WideResNet', #'ResNet18',
                         input_shape=(1, 3, 32, 32),
                         teacher_path='results/No-curr_distil/cifar10/WideResNet_p0/model.pt'),

              ckpt=dict(is_load=False,
                        is_save=True,
                        is_save_pic=False,
                        dir='results/',
                        save_every=10),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.1,
                             weight_decay=5e-4),

              scheduler=dict(type="cosine_annealing",
                             T_max=250),

              ds_strategy=dict(type="RewLam",
                               warm_epoch=10,
                               select_every=10,
                               decay=0.2,
                               schedule=[0, 10, 20, 40, 60, 100, 140, 170, 200, 251],
                               sch_ind=1),

              train_args=dict(num_epochs=250,
                              device="cuda",
                              print_every=2,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time", "trn_loss", "trn_acc"],
                              return_args=[]
                              )
              )

'''
dss_strategy=dict(type="GradMatch",
               fraction=0.1,
               select_every=20,
               lam=0.5,
               valid=False),

teacher_path = 'results/No-curr_class/cifar100/WideResNet_f/model.pt'''

