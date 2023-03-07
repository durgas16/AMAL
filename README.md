# Adaptive Mixing of Auxiliary Losses in Supervised Learning
Durga Sivasubramanian, Ayush Maheshwari, Prathosh AP, Pradeep Shenoy, Ganesh Ramakrishnan

[![arXiv](https://img.shields.io/badge/arXiv-2106.02584-b31b1b.svg)](https://arxiv.org/abs/2202.03250)

## Required packages
- pytorch (GPU preferrable)
- python >= 3.6

## Running the Code For KD
The command to run the code is `python3 train_distill.py` to perform knowledge distillation with clean data and  `python3 train_distill_noise.py` to perform knowledge distillation in the noisy setting. </br>

To train a *ResNet56* as a teacher model, set pass `--lam 0.0` and set architecture in *config_no_amal_\** config file as Resnet56 i.e. `model=dict(architecture='ResNet56',...) `. </br>

To run standard knowledge distillation experiment pass appropriate lambda value and set `architecture` in *config_no_amal\** config file as student architecture, set `teacher_arch` to the teacher architecture and set path to tranined teacher model using `teacher_path` key i.e `model=dict(architecture='ResNet8',...,teacher_arch=['ResNet110'],...,teacher_path=['results/.../model.pt'])`. </br>

To run knowledge distillation with amal, pass *config_amal\** file  with approriate student architecture, teacher architecture and path to tranined teacher model.

Each experiment will produce a log file with the loss/accuracy in the results folder along with latest checkpoint which has latest model, lambda values etc using which results in the main paper can be reproduced.</br>
