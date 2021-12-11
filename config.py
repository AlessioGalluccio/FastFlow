'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''
import torch

# device settings
#'cuda' or 'cpu'
device = 'cpu'

if device == 'cuda':
    torch.cuda.set_device(0)

# neptune
neptune_activate = False

# data settings
dataset_path = "dummy_dataset"
class_name = "dummy_class"
modelname = "hazelnut_test"


# transformation settings
transf_rotations = True
transf_brightness = 0.0
transf_contrast = 0.0
transf_saturation = 0.0
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# feature extractor
# select "resnet18", "deit", or "cait"
extractor_name = "cait"

# network hyperparameters
n_scales = 1 # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
clamp_alpha = 3 # see paper (differnet) equation 2 for explanation
n_coupling_blocks = 4
#fc_internal = 2048 # number of neurons in hidden layers of s-t-networks
dropout = 0.0 # dropout in s-t-networks
lr_init = 2e-4
subnet_conv_dim = 128 # internal dimension of the convolutional layera

if(extractor_name == "resnet18"):
    n_feat = 64*64*64*n_scales
    img_size = (256, 256)
elif(extractor_name == "deit"):
    n_feat = 24*24*768*n_scales
    img_size = (384, 384)
elif(extractor_name == "cait"):
    n_feat = 28*28*768*n_scales
    img_size = (448, 448)
else:
    n_feat = 256 * n_scales # do not change except you change the feature extractor
    img_size = (448, 448)

img_dims = [3] + list(img_size)

# dataloader parameters
n_transforms = 4 # number of transformations per sample in training
n_transforms_test = 64 # number of transformations per sample in testing
batch_size = 24 # actual batch size is this value multiplied by n_transforms(_test)
batch_size_test = batch_size * n_transforms // n_transforms_test

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 24
sub_epochs = 8

# output settings
verbose = True
grad_map_viz = False
hide_tqdm_bar = False
save_model = True

