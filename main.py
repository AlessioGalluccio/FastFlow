'''This is the repo which contains the original code to the WACV 2021 paper
"Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''

import config as c
from train import train
from utils import load_datasets, make_dataloaders
import os

import sys
print("Python version")
print (sys.version)
print(os.listdir())
print(os.listdir("./data"))

#I change the location where pytorch saves pretrained models
os.environ['TORCH_HOME'] = 'models\\alexnet' #setting the environment variable

#import torch
#xcv = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained=True)

#manage dataset
from handledata import handledata
#handledata()

train_set, test_set = load_datasets(c.dataset_path, c.class_name)
train_loader, test_loader = make_dataloaders(train_set, test_set)
model = train(train_loader, test_loader)
