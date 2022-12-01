#LearninNet110', setting
config = dict(setting="supervisedlearning",

              dataset=dict(name="cifar100",
                           datadir="../data",
                           feature="dss",
                           type="pre-defined",
                           grad_fit=2),

              dataloader=dict(shuffle=True,
                              trn_batch_size=128,
                              val_batch_size=250,
                              tst_batch_size=1000,
                              pin_memory=True),

              model=dict(architecture='Wide_16_2',#'ShuffleV1',#'MobileNetV2',#'vgg8',#'ResNet50',
                         numclasses=100,
                         teacher_arch='Wide_40_2', #['ResNet50'], #
                         depth_teach = [10],
                         depth = 2,
                         teacher_path='/home/suraksha/SSKD/experiments/wrn_40_2.pth'),
                         #['results/No-curr_distil/cifar100/ResNet50_p0/24/model.pt']),
                         #['results/No-curr_distil/cifar100/ResNet110_p0/24/model.pt']),
                         
              
              ckpt=dict(is_load=False,
                        is_save=True,
                        is_save_pic=False,
                        dir='results/',
                        save_every=10),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.05,
                             weight_decay=5e-4),

              scheduler=dict(type="Mstep",
                             T_max=240),

              ds_strategy=dict(type="No-curr",#"MultiLam",
                               warm_epoch=10,
                               select_every=10,
                               decay=0.2,
                               schedule=[0, 10, 20, 40, 60, 100, 140, 160,240],
                               sch_ind=1),

              train_args=dict(num_epochs=240,
                              device="cuda",
                              print_every=1,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time", "trn_loss", "trn_acc"],
                              return_args=[]
                              )
              )


