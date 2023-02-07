# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from models.utils.continual_model import ContinualModel
import torchvision.transforms as transforms
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
import wandb


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via Lagrangian Duality')
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
        self.task = 0

    def begin_task(self, dataset):

        self.lambdas = torch.ones(len(dataset.train_loader.dataset), requires_grad=False, device = self.device)
        self.buf_lambdas = torch.ones(self.args.buffer_size, requires_grad=False, device = self.device)

    def end_task(self, dataset):
        
        idxs_lambdas_descending = (-self.lambdas).argsort()
        samples_per_task = self.args.buffer_size // dataset.N_TASKS # e.g: 200 with bufsize = 1000
        samples_per_class = self.args.buffer_size // (2*(self.task+1)) # e.g: 100
        num_outliers = 100


        if self.buffer.is_empty():

            assert self.buffer.buffer_size + num_outliers < len(idxs_lambdas_descending) , "Tried to add duplicates to buffer."

            imgs, targets = [], []

            space_per_class = np.zeros(dataset.N_CLASSES_PER_TASK*dataset.N_TASKS) 
            space_per_class[:dataset.N_CLASSES_PER_TASK] += samples_per_class

            for i in idxs_lambdas_descending[num_outliers:]:
                img, target, original_img, index = dataset.train_loader.dataset[i]
                if space_per_class[target] > 0:
                    imgs.append(original_img.unsqueeze(0))
                    targets.append(target)
                    space_per_class[target] -= 1
                if np.sum(space_per_class) == 0:
                    break 
            
            self.buffer.add_data(
                examples=torch.cat(imgs),
                labels=torch.Tensor(targets),
            )
        
        else:

            # Get all buffer
            all_data = self.buffer.get_all_data(transform = None)

            # Empty old buffer to replace with new one
            self.buffer.empty()

            # Get samples from new task
            imgs, targets = [], []
            
            space_per_class = np.zeros(dataset.N_CLASSES_PER_TASK*dataset.N_TASKS) 
            space_per_class[self.task*dataset.N_CLASSES_PER_TASK:self.task*dataset.N_CLASSES_PER_TASK+dataset.N_CLASSES_PER_TASK] += samples_per_class

            for i in idxs_lambdas_descending[num_outliers:]:
                img, target, original_img, index = dataset.train_loader.dataset[i]
                if space_per_class[target] > 0:
                    imgs.append(original_img.unsqueeze(0))
                    targets.append(target)
                    space_per_class[target] -= 1
                if np.sum(space_per_class) == 0:
                    break     

            self.buffer.add_data(
                examples=torch.cat(imgs),
                labels=torch.Tensor(targets),
            )

            # Get samples from buffer randomly but uniformly
            examples, labels = all_data
            random_idxs = np.arange(len(labels))
            np.random.shuffle(random_idxs)
            
            imgs, targets = [], []

            space_per_class = np.zeros(dataset.N_CLASSES_PER_TASK*dataset.N_TASKS) 
            space_per_class[:self.task*dataset.N_CLASSES_PER_TASK] += samples_per_class

            for i in random_idxs:
                if space_per_class[labels[i]] > 0:
                    imgs.append(examples[i].unsqueeze(0))
                    targets.append(labels[i])
                    space_per_class[labels[i]] -= 1
                if np.sum(space_per_class) == 0:
                    break     

            self.buffer.add_data(
                examples=torch.cat(imgs),
                labels=torch.Tensor(targets),
            )

        self.buffer.create_loader(size = self.args.minibatch_size, model_transform = self.transform)
        self.task += 1
                
    def observe(self, inputs, labels, not_aug_inputs, indexes):

        real_batch_size = inputs.shape[0]
        lambdas = self.lambdas[indexes]

        self.opt.zero_grad()
        
        if not self.buffer.is_empty():

            #buf_indexes, buf_inputs, buf_labels = self.buffer.get_data(
            #    self.args.minibatch_size, transform=self.transform, return_index=True)
            buf_indexes, buf_inputs, buf_labels = self.buffer.get_data_from_loader(size = self.args.minibatch_size)
            
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
