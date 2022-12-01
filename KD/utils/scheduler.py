from os import stat_result
import numpy as np


class Scheduler:

    def __init__(self, optimizer, lr, lr_max, lr_max_decay, lr_min, lr_min_decay, schedule):

        self.lr = lr
        self.lr_max = lr_max
        self.lr_max_decay = lr_max_decay
        self.lr_min = lr_min
        self.lr_min_decay = lr_min_decay
        self.sch_ind = 1
        self.schedule = schedule
        self.total_steps = schedule[self.sch_ind + 1] - schedule[self.sch_ind]
        self.optimizer = optimizer

    def adjust_cosine_learning_rate_step(self, epoch):

        # print(self.total_steps,self.lr_max,self.lr_min)
        if epoch >= self.schedule[self.sch_ind]:

            if epoch == self.schedule[self.sch_ind + 1]:
                self.sch_ind += 1
                self.total_steps = self.schedule[self.sch_ind + 1] - self.schedule[self.sch_ind]

                self.lr_max *= self.lr_max_decay
                self.lr_min *= self.lr_min_decay

            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (
                        1 + np.cos((float(epoch - self.schedule[self.sch_ind]) / float(self.total_steps)) * np.pi))

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def load_state_dict(self, state):
        self.lr = state['lr']
        self.lr_max = state['lr_max']
        self.lr_max_decay = state['lr_max_decay']
        self.lr_min = state['lr_min']
        self.lr_min_decay = state['lr_min_decay']
        self.sch_ind = state['sch_ind']
        self.schedule = state['schedule']
        self.total_steps = state['total_steps']

    def state_dict(self):
        state = {}
        state['lr'] = self.lr
        state['lr_max'] = self.lr_max
        state['lr_max_decay'] = self.lr_max_decay
        state['lr_min'] = self.lr_min
        state['lr_min_decay'] = self.lr_min_decay
        state['sch_ind'] = self.sch_ind
        state['schedule'] = self.schedule
        state['total_steps'] = self.total_steps

        return state


