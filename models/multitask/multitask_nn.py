import sys
import pathlib
import pdb
from typing import List
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass

class MultiTaskNNSep_v2(nn.Module):
    def __init__(self, l_sizes=[(16,8,4), (16,8,4)], class_op_size=22) -> None:
        '''
        Architecture where classifier and regressor do not share layers (essentially 2 separate NNs). Loading prediction is binned.

        Args:
            l_sizes: An iterable of length 2. The first entry contains a list of layers sizes for the classifier, the second entry for the regressor. Note the first entry in both lists is the input feature size.
        '''
        #TODO: I do not see any difference between this and v1 of the model except for the softmax... 
        super().__init__()
        self.c_sizes, self.r_sizes = l_sizes
        self.class_op_size = class_op_size

        # classifier
        c_num_layers = len(self.c_sizes)
        c_layers_list = [
            nn.Sequential(
                nn.Linear(self.c_sizes[i], self.c_sizes[i+1]), 
                nn.ReLU()
            ) for i in range(0, c_num_layers-1) 
        ]
        self.classifier = nn.Sequential(
            *c_layers_list,
            # nn.Linear(c_sizes[-1], 1)
            nn.Linear(self.c_sizes[-1], self.class_op_size),
            nn.Softmax(dim=1) 
        )

        # regressor
        r_num_layers = len(self.r_sizes)
        r_layers_list = [
            nn.Sequential(
                nn.Linear(self.r_sizes[i], self.r_sizes[i+1]),
                nn.ReLU()
            ) for i in range(0, r_num_layers-1) 
        ]
        self.regressor = nn.Sequential(
            *r_layers_list,
            nn.Linear(self.r_sizes[-1], 1)
        )
    
    def forward(self, x):
        return self.classifier(x), self.regressor(x)


class MultiTaskNNCorr_v2(nn.Module): 
    def __init__(self, l_sizes=[(16,8), (8,4), (8,4)], class_op_size=22) -> None:
        '''
        Architecture where classifier and regressor share layers. Loading prediction is binned.

        Args:
            l_sizes: An iterable of length 2. The first entry contains a lsit of layer sizes for the common layers, the second entry contains a list of layers sizes for the classifier, the third entry for the regressor. Note the first entry in both lists is the input feature size.
        '''
        super().__init__()
        self.com_sizes, self.c_sizes, self.r_sizes = l_sizes
        self.class_op_size = class_op_size
        # check common output layer is same size as input layers for both classifier and regressor
        assert self.c_sizes[0] == self.r_sizes[0]

        # common
        com_num_layers = len(self.com_sizes)
        com_layers_list = [
            nn.Sequential(
                nn.Linear(self.com_sizes[i], self.com_sizes[i+1]), 
                nn.ReLU()
            ) for i in range(0, com_num_layers-1) 
        ]
        self.common = nn.Sequential(
            *com_layers_list,
            # nn.Linear(com_sizes[-1], c_sizes[0]) 
        )

        # classifier
        c_num_layers = len(self.c_sizes)
        c_layers_list = [
            nn.Sequential(
                nn.Linear(self.c_sizes[i], self.c_sizes[i+1]), 
                nn.ReLU()
            ) for i in range(0, c_num_layers-1) 
        ]
        self.classifier = nn.Sequential(
            self.common,
            *c_layers_list,
            # nn.Linear(c_sizes[-1], 1)
            nn.Linear(self.c_sizes[-1], self.class_op_size),
            nn.Softmax(dim=1) 
        )

        # regressor
        r_num_layers = len(self.r_sizes)
        r_layers_list = [
            nn.Sequential(
                nn.Linear(self.r_sizes[i], self.r_sizes[i+1]),
                nn.ReLU()
            ) for i in range(0, r_num_layers-1) 
        ]
        self.regressor = nn.Sequential(
            self.common,
            *r_layers_list,
            nn.Linear(self.r_sizes[-1], 1)
        )

    def forward(self, x):
        return self.classifier(x), self.regressor(x)


class MultiTaskNNSep(nn.Module):
    def __init__(self, l_sizes=[(16,8,4), (16,8,4)], class_op_size=1) -> None:
        '''
        Architecture where classifier and regressor do not share layers (essentially 2 separate NNs). Loading prediction is regression.

        Args:
            l_sizes: An iterable of length 2. The first entry contains a list of layers sizes for the classifier, the second entry for the regressor. Note the first entry in both lists is the input feature size.
        '''
        super().__init__()
        self.c_sizes, self.r_sizes = l_sizes
        self.class_op_size = class_op_size

        # classifier
        c_num_layers = len(self.c_sizes)
        c_layers_list = [
            nn.Sequential(
                nn.Linear(self.c_sizes[i], self.c_sizes[i+1]), 
                nn.ReLU()
            ) for i in range(0, c_num_layers-1) 
        ]
        self.classifier = nn.Sequential(
            *c_layers_list,
            nn.Linear(self.c_sizes[-1], self.class_op_size)

        )
        # regressor
        r_num_layers = len(self.r_sizes)
        r_layers_list = [
            nn.Sequential(
                nn.Linear(self.r_sizes[i], self.r_sizes[i+1]),
                nn.ReLU()
            ) for i in range(0, r_num_layers-1) 
        ]
        self.regressor = nn.Sequential(
            *r_layers_list,
            nn.Linear(self.r_sizes[-1], 1)
        )
    
    def forward(self, x):
        return self.classifier(x), self.regressor(x)


class MultiTaskNNCorr(nn.Module): 
    def __init__(self, l_sizes=[(16,8), (8,4), (8,4)], class_op_size=1) -> None:
        '''
        Architecture where classifier and regressor share layers. Loading prediction is regression.

        Args:
            l_sizes: An iterable of length 2. The first entry contains a lsit of layer sizes for the common layers, the second entry contains a list of layers sizes for the classifier, the third entry for the regressor. Note the first entry in both lists is the input feature size.
        '''
        super().__init__()
        self.com_sizes, self.c_sizes, self.r_sizes = l_sizes
        self.class_op_size = class_op_size
        # check common output layer is same size as input layers for both classifier and regressor
        assert self.c_sizes[0] == self.r_sizes[0]

        # common
        com_num_layers = len(self.com_sizes)
        com_layers_list = [
            nn.Sequential(
                nn.Linear(self.com_sizes[i], self.com_sizes[i+1]), 
                nn.ReLU()
            ) for i in range(0, com_num_layers-1) 
        ]
        self.common = nn.Sequential(
            *com_layers_list,
            # nn.Linear(com_sizes[-1], c_sizes[0]) 
        )

        # classifier
        c_num_layers = len(self.c_sizes)
        c_layers_list = [
            nn.Sequential(
                nn.Linear(self.c_sizes[i], self.c_sizes[i+1]), 
                nn.ReLU()
            ) for i in range(0, c_num_layers-1) 
        ]
        self.classifier = nn.Sequential(
            self.common,
            *c_layers_list,
            nn.Linear(self.c_sizes[-1], self.class_op_size)
        )

        # regressor
        r_num_layers = len(self.r_sizes)
        r_layers_list = [
            nn.Sequential(
                nn.Linear(self.r_sizes[i], self.r_sizes[i+1]),
                nn.ReLU()
            ) for i in range(0, r_num_layers-1) 
        ]
        self.regressor = nn.Sequential(
            self.common,
            *r_layers_list,
            nn.Linear(self.r_sizes[-1], 1)
        )

    def forward(self, x):
        return self.classifier(x), self.regressor(x)


MULTITASK_MODELS = {
    'multitasknnsep': MultiTaskNNSep, 
    'multitasknncorr': MultiTaskNNCorr,
    'multitasknnsep_v2': MultiTaskNNSep_v2, 
    'multitasknncorr_v2': MultiTaskNNCorr_v2,
    }


class OldMultiTaskNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # hidden n-1 layers
        l_sizes = [int(x) for x in l_sizes]
        num_layers = len(l_sizes)
        hl_list = [
            nn.Sequential(
                nn.Linear(l_sizes[i], l_sizes[i+1]),
                nn.ReLU()
            ) for i in range(0, num_layers-1) # TODO: check
        ]
        self.hl = nn.Sequential(*hl_list)

        # final hidden layer
        fl_dict = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
        self.fl = nn.Sequential(
            nn.Linear(l_sizes[-2], l_sizes[-1]),
            fl_dict[fnl]
            )

        # output layer
        self.ol = nn.Sequential(nn.Linear(l_sizes[-1],2))

        # all
        self.net = nn.Sequential(self.hl, self.fl, self.ol)
        print(self.net)


    def forward(self, x):

        return self.net(x)


# testing
if __name__ == '__main__':
    # model = MultiTaskNNSep()
    model = MultiTaskNNCorr()
    x = torch.randn(16)
    op = model(x)
    print(x) 