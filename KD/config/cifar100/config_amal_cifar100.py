# Learning setting
config = dict(setting="supervisedlearning",

              dataset=dict(name="cifar100",
                           datadir="../data",
                           feature="dss",
                           type="pre-defined",
                           grad_fit=3),

              dataloader=dict(shuffle=True,
                              trn_batch_size=128,
                              val_batch_size=1000,
                              tst_batch_size=1000,
                              pin_memory=True),

              model=dict(architecture='ResNet8', 
                         numclasses=100,
                         teacher_arch=['ResNet56'], 
                         depth_teach = [16],
                         width_teach = [3],
                         depth = 16,
                         width = 1,
                         teacher_path=['results/No-curr_distilT/cifar100/ResNet56_p0.0/4/0/24/model.pt']),

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

              scheduler=dict(type="Mstep",
                             T_max=200),

              ds_strategy=dict(type="MultiLam", #'LearnLam',#
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
'''model=dict(architecture='WRN_16_X', 
                         numclasses=100,
                         teacher_arch=['WRN_16_X'], 
                         depth_teach = [16],
                         width_teach = [3],
                         depth = 16,
                         width = 1,
                         teacher_path=['results/No-curr_distil/cifar100/WRN_16_X_16_3_p0/16/model.pt']),'''