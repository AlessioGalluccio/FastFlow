import os
import torch
from torch import nn
from torchsummary import summary

import config as c
import FrEIA.modules as Fm
import FrEIA.framework as Ff

import torchvision.models as models

import numpy as np

WEIGHT_DIR = './weights'
MODEL_DIR = './models'

def subnet_conv_1(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c.subnet_conv_dim,   kernel_size=(1,1), padding='same'),
                         nn.ReLU(),
                         nn.Conv2d(c.subnet_conv_dim,  c_out, kernel_size=(1,1), padding='same'))

def subnet_conv_3(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c.subnet_conv_dim,   kernel_size=(3,3), padding='same'),
                         nn.ReLU(),
                         nn.Conv2d(c.subnet_conv_dim,  c_out, kernel_size=(3,3), padding='same'))


def nf_fast_flow(input_dim):
    nodes = list()

    nodes.append(Ff.InputNode(input_dim[0],input_dim[1], input_dim[2], name='input'))
    # I add blocks with 3x3 and 1x1 convolutions alternatively. Before them, I add a fixed permutation of the channels
    for k in range(c.n_coupling_blocks):
        # It permutes the first dimension, the channels
        '''
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':k},
                             name=F'permute_high_res_{k}'))
        '''
        if k % 2 == 0 or c.only_3x3_convolution:
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.AllInOneBlock,
                                 {'subnet_constructor':subnet_conv_3, 'affine_clamping':c.clamp},
                                 name=F'conv_high_res_{k}'))
        else:
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.AllInOneBlock,
                                 {'subnet_constructor':subnet_conv_1, 'affine_clamping':c.clamp},
                                 name=F'conv_high_res_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    #print(nodes)
    coder = Ff.GraphINN(nodes)
    #print(coder)
    return coder


class FastFlow(nn.Module):
    def __init__(self):
        super(FastFlow, self).__init__()

        if c.extractor_name == "resnet18":
            self.feature_extractor = models.resnet18(pretrained=True)
            # I take only the first blocks of the net, which has 64x64x64 as output
            self.feature_extractor = torch.nn.Sequential(*(list(self.feature_extractor.children())[:5]))

            # freeze the layers
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

            self.feature_extractor.to(c.device)
            print(summary(self.feature_extractor, (3,256,256), device=c.device))
            #self.feature_extractor = torch.load('./pretrained/M48_448.pth') #sbagliato, carica solo i pesi, non il modello
            #self.feature_extractor.eval() # to deactivate the dropout layers

            # This input is unfortunately hardcoded. See the output dimensions of the feature extractor.
            # Don't add the batch size (first number)
            self.nf = nf_fast_flow((64,64,64))

        elif c.extractor_name == "deit":
            self.feature_extractor = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_384', pretrained=True)
            # I select the input layers and the first 7 blocks
            self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:2],
                                                         *list(list(self.feature_extractor.children())[2].children())[:7])
            self.feature_extractor.to(c.device)
            # freeze the layers
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print(summary(self.feature_extractor, (3,384,384), device=c.device))
            self.nf = nf_fast_flow((24,24,768))

        elif c.extractor_name == "cait":
            self.feature_extractor = torch.hub.load('facebookresearch/deit:main', 'cait_M48', pretrained=True)
            self.feature_extractor.to(c.device)

            # how to print the first 5 Layerscale blocks (input layers are not included
            print(list(list(self.feature_extractor.children())[2].children())[:5])

            # this network has a gigantic children called ModuleList, that's why we can't use only children() method to split the network
            # ModuleList contains many Layerscale blocks. We want to select only the first 20 ones
            # ModuleList content can be viewed in list(self.feature_extractor.children())[2]
            self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:2],
                                                         *list(list(self.feature_extractor.children())[2].children())[:20])


            # freeze the layers
            for param in self.feature_extractor.parameters():
                param.requires_grad = False



            print(summary(self.feature_extractor, (3,448,448), device=c.device))
            self.nf = nf_fast_flow((28,28,768))

    def forward(self, x):
        feat_s = self.feature_extractor(x)

        # I have to reshape the linearized output of deit back to a 2D image
        # From (576,768) to (24,24,768). The first number is the batch size
        if c.extractor_name == "deit":
            dim_batch = feat_s.size(dim=0)
            feat_s = feat_s.reshape(dim_batch,24,24,768)
            #print(feat_s.size())

        # I have to reshape the linearized output of cait back to a 2D image
        if c.extractor_name == "cait":
            dim_batch = feat_s.size(dim=0)
            feat_s = feat_s.reshape(dim_batch,28,28,768)

        # Resnet doesn't need reshape

        z, log_jac_det = self.nf(feat_s)
        return z, log_jac_det



def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path)
    return model


def save_weights(model, filename):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, filename))


def load_weights(model, filename):
    path = os.path.join(WEIGHT_DIR, filename)
    model.load_state_dict(torch.load(path))
    return model
