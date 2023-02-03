# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
import wandb


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class cally(ContinualModel):
   
    NAME = 'cally'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        
        super(cally, self).__init__(backbone, loss, args, transform)
        
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.epsilon = 0.1
        self.lr_dual = 0.5

    def begin_task(self, dataset):

        self.lambdas = torch.zeros(len(dataset.train_loader.dataset), requires_grad=False, device = self.device)
        self.buf_lambdas = torch.zeros(self.args.buffer_size, requires_grad=False, device = self.device)

    def end_task(self, dataset):
        
        idxs_lambdas_descending = (-self.lambdas).argsort()
        samples_per_task = self.args.buffer_size // dataset.N_TASKS
        num_outliers = 30

        imgs, targets = [], []
        for i in idxs_lambdas_descending[num_outliers:samples_per_task]:

            img, target, original_img, index = dataset.train_loader.dataset[i]
            imgs.append(original_img)
            targets.append(target)

        self.buffer.add_data(
            examples=torch.cat(imgs),
            labels=torch.Tensor(targets),
        )

        self.task += 1
                
    def observe(self, inputs, labels, not_aug_inputs, indexes):

        real_batch_size = inputs.shape[0]
        lambdas = self.lambdas[indexes]

        self.opt.zero_grad()
        
        if not self.buffer.is_empty():

            buf_indexes, buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True)
            
            buf_lambdas = self.buf_lambdas[buf_indexes]

            # Forward Pass
            outputs = self.net(inputs)
            buf_outputs = self.net(buf_inputs)
            
            # Evaluate Lagrangian
            loss = self.loss(outputs, labels, reduction = 'none')
            buf_loss = self.loss(buf_outputs, buf_labels.squeeze(), reduction = 'none')
            lagrangian = torch.mean(lambdas*(loss-self.epsilon)) + torch.mean(buf_lambdas*(buf_loss - self.epsilon))
            
            # Primal Update
            lagrangian.backward()
            self.opt.step()

            # Dual Update
            lambdas += self.lr_dual*(loss - self.epsilon)
            lambdas = torch.nn.ReLU()(lambdas)
            self.lambdas[indexes] = lambdas.detach()

            buf_lambdas += self.lr_dual*(buf_loss - self.epsilon)
            buf_lambdas = torch.nn.ReLU()(buf_lambdas)
            self.buf_lambdas[buf_indexes] = buf_lambdas.detach()

        else:

            # Forward Pass
            outputs = self.net(inputs)
            
            # Evaluate Lagrangian
            loss = self.loss(outputs, labels, reduction = 'none')
            lagrangian = torch.mean(lambdas*(loss - self.epsilon))

            # Primal Update
            lagrangian.backward()
            self.opt.step()

            # Dual Update
            lambdas += self.lr_dual*(loss - self.epsilon)
            lambdas = torch.nn.ReLU()(lambdas)
            self.lambdas[indexes] = lambdas.detach()

        return lagrangian.item()
