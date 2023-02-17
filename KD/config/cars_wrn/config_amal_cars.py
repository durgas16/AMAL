config = dict(setting="supervisedlearning",

              dataset=dict(name="cars",
                           datadir="../data",
                           feature="dss",
                           type="pre-defined",
                           grad_fit=2),

              dataloader=dict(shuffle=True,
                              trn_batch_size=64,
                              val_batch_size=64,
                              tst_batch_size=128,
                              pin_memory=True),

             model=dict(architecture='WRN_16_X',
                         numclasses=196,
                         teacher_arch=['WRN_16_X'], 
                         depth_teach = [16],
                         width_teach = [8],
                         depth = 16,
                         width = 1,
                         teacher_path=['results/No-curr_distil/cars/WRN_16_X_16_8_p0/16/model.pt']),

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

              ds_strategy=dict(type="MultiLam",
                               warm_epoch=10,
                               select_every=10,
                               ),

              train_args=dict(num_epochs=200,
                              device="cuda",
                              print_every=5,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time", "trn_loss", "trn_acc"],
                              return_args=[]
                              )
              )
