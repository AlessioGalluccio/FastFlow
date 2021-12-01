import os
import torch
from torch import nn
from torchsummary import summary

import config as c
from freia_funcs import permute_layer, glow_coupling_layer, F_fully_connected, ReversibleGraphNet, OutputNode, \
    InputNode, Node
import FrEIA.modules as Fm
import FrEIA.framework as Ff

import torchvision.models as models

WEIGHT_DIR = './weights'
MODEL_DIR = './models'

def subnet_conv_1(c_in, c_out, kernel_size=1):
    return nn.Sequential(nn.Conv2d(c_in, c.subnet_conv_dim,   kernel_size=(kernel_size,kernel_size), padding='same'),
                         nn.ReLU(),
                         nn.Conv2d(c.subnet_conv_dim,  c_out, kernel_size=(kernel_size,kernel_size), padding='same'))

def subnet_conv_3(c_in, c_out, kernel_size=3):
    return nn.Sequential(nn.Conv2d(c_in, c.subnet_conv_dim,   kernel_size=(kernel_size,kernel_size), padding='same'),
                         nn.ReLU(),
                         nn.Conv2d(c.subnet_conv_dim,  c_out, kernel_size=(kernel_size,kernel_size), padding='same'))


def nf_head(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamp_alpha, 'F_class': F_fully_connected,
                           'F_args': {'internal_size': c.fc_internal, 'dropout': c.dropout}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder

def nf_fast_flow(input_dim):
    nodes = list()

    nodes.append(Ff.InputNode(input_dim[0],input_dim[1], input_dim[2], name='input'))
    for k in range(c.n_coupling_blocks):
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':k},
                             name=F'permute_high_res_{k}'))
        if k % 2 == 0:
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet_conv_3, 'clamp':1.2},
                                 name=F'conv_high_res_{k}'))
        else:
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet_conv_1, 'clamp':1.2},
                                 name=F'conv_high_res_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    print(nodes)
    coder = Ff.GraphINN(nodes)
    print(coder)
    return coder


class FastFlow(nn.Module):
    def __init__(self):
        super(FastFlow, self).__init__()

        #self.feature_extractor = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained=True)
        #self.feature_extractor = torch.nn.Sequential(*(list(self.feature_extractor.children())[:-2])) # I remove the last two layers

        self.feature_extractor = models.resnet18()
        self.feature_extractor = torch.nn.Sequential(*(list(self.feature_extractor.children())[:5]))

        print(summary(self.feature_extractor, (3,224,224)))
        #self.feature_extractor = torch.load('./pretrained/M48_448.pth') #sbagliato, carica solo i pesi, non il modello
        #self.feature_extractor.eval() # to deactivate the dropout layers
        self.nf = nf_fast_flow((64,56,56))

    def forward(self, x):
        y_cat = list()

        '''
        for s in range(c.n_scales):
            x_scaled = F.interpolate(x, size=c.img_size[0] // (2 ** s)) if s > 0 else x
            #feat_s = self.feature_extractor.features(x_scaled)
            feat_s = self.feature_extractor(x_scaled)
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))
        '''
        feat_s = self.feature_extractor(x)
        #y_cat.append(feat_s)
        #y = torch.cat(y_cat, dim=3)
        #print(feat_s)
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
