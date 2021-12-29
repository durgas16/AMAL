# Learning setting
config = dict(setting="supervisedlearning",

              dataset=dict(name="synthetic",
                           datadir="../data",
                           feature="dss",
                           grad_fit=30,
                           grad_batch=1,
                           type="blob",#class",
                           test_type="None"),

              dataloader=dict(shuffle=True,
                              trn_batch_size=128,
                              val_batch_size=1000,
                              tst_batch_size=1000,
                              pin_memory=True),

              model=dict(architecture='NN_2L',
                         type='pre-defined',
                         numclasses=8,
                         input_dims=2,#14,
                         hidden_units_teach=5,
                         hidden_units_stu=2,
                         teacher_arch='NN_2L',  # 'ResNet18',
                         teacher_path='results/No-curr_distil/synthetic/NN_2L_5_p0/blob/None/16/model.pt'),

              ckpt=dict(is_load=False,
                        is_save=False,
                        is_save_pic=True,
                        dir='results/',
                        save_every=5),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.01,
                             weight_decay=5e-4),

              scheduler=dict(type="cosine_annealing",
                             T_max=101),

              ds_strategy=dict(type="LearnLam",
                            warm_epoch =0,
                            select_every=10,
                            decay = 0.2,
                            schedule=[0,10,40,101],#151],#250,301],
                            sch_ind = 1),

              train_args=dict(num_epochs=100,
                              device="cuda",
                              print_every=5,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time","trn_loss","trn_acc"],
                              return_args=[]
                              )
              )
