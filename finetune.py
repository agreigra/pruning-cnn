import torch
from torch.autograd import Variable
from torchvision import models
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import time
from heapq import nsmallest
from operator import itemgetter

class FilterPrunner:
    def __init__(self, model, device):
        self.model = model
        self.reset()
        self.device = device
    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1
        
        if(self.model.avgpool):
            x = self.model.avgpool(x)
        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data


        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.device)

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i]).cpu()
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune             

class PrunningFineTuner:
    def __init__(self, train_path, test_path, model, device, size):
        self.train_data_loader = dataset.loader(train_path, size = size)
        self.test_data_loader = dataset.test_loader(test_path, size = size)
        self.device = device
        self.device = device
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model, device) 
        

    def test(self):
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            
            batch = batch.to(self.device)
            output = self.model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)
        return float(correct) / total

    def train(self, optimizer = None, epoches=10):
        if optimizer is None:
            optimizer = optim.SGD(self.model.classifier.parameters(), lr=0.0001, momentum=0.9)

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            print("Accuracy :",self.test())
        print("Finished fine tuning.")
        

    def train_batch(self, optimizer, batch, label, rank_filters):

        
        batch = batch.to(self.device)
        label = label.to(self.device)

        self.model.zero_grad()
        batch = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(batch)
            
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(batch), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer = None, rank_filters = False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters = True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)
        
    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self, filter_to_prune):
        
        #Accuracy befure pruning 
        print("Accuracy before pruning:",self.test())
        #Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune = filter_to_prune

        print("Number of filters to prune", num_filters_to_prune)

        print("Ranking filters.. ")
        prune_targets = self.get_candidates_to_prune(num_filters_to_prune)
        layers_prunned = {}
        for layer_index, filter_index in prune_targets:
            if layer_index not in layers_prunned:
                layers_prunned[layer_index] = 0
            layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

        print("Layers that will be prunned", layers_prunned)
        print("Prunning filters.. ")
        model = self.model.cpu()
        for layer_index, filter_index in prune_targets:
            model = prune_conv_layer(model, layer_index, filter_index, self.device)

        self.model = model.to(self.device)
        message = str(100*float((number_of_filters - self.total_num_filters())) / number_of_filters) + "%"
        print("Filters prunned", str(message))
        acc = self.test()
        print("Accuracy after pruning:",acc)
        print("Retrain to recover from prunning iteration.")
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.train(optimizer, epoches = 3)
        return acc
        
    def get_number_of_parameters(self):
        
        return sum(p.numel() for p in self.model.parameters())
    