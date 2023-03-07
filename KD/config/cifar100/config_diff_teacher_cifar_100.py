# Learning setting
config = dict(setting="supervisedlearning",

              dataset=dict(name="cifar100",
                           datadir="../data",
                           feature="dss",
                           type="pre-defined",
                           grad_fit=4),

              dataloader=dict(shuffle=True,
                              trn_batch_size=128,
                              val_batch_size=1000,
                              tst_batch_size=1000,
                              pin_memory=True),

             model=dict(architecture='ResNet8', #'DenseNet_X',#
                         numclasses=100,
                         teacher_arch=['ResNet14','ResNet26','ResNet32','ResNet56'], 
                         
                         teacher_path=['results/No-curr_distil/cifar100/ResNet14_p0/24/model.pt',\
                         'results/No-curr_distil/cifar100/ResNet26_p0/24/model.pt',\
                         'results/No-curr_distil/cifar100/ResNet32_p0/24/model.pt',\
                         'results/No-curr_distil/cifar100/ResNet56_p0/24/model.pt']),

              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=10),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.1,
                             weight_decay=5e-4),

              scheduler=dict(type="cosine_annealing",#"Mstep",
                             T_max=201),

              ds_strategy=dict(type="MultiLam",#"No-curr"
                               warm_epoch=10,
                               select_every=10
                               ),

              train_args=dict(num_epochs=200,
                              device="cuda",
                              print_every=2,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time", "trn_loss", "trn_acc"],
                              return_args=[]
                              )
              )
