import cfgs.config as cfg_p
import torch
import torch.nn as nn
import torchvision
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import math

import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import os

from PIL import Image
import cfgs.config as cfg
import scipy.misc 
import cv2
import matplotlib.pyplot as plt

def create_patch_mask(in_features, my_patch, patch_size):
    width = in_features.size(1)
    height = in_features.size(2)
    patch_mask = torch.zeros([3, width,height])

    p_w = patch_size + cfg_p.patch_x
    p_h = patch_size + cfg_p.patch_y
    patch_mask[:, int(cfg_p.patch_x):int(p_w), int(cfg_p.patch_y):int(p_h)]= 1

    return patch_mask

def create_img_mask(in_features, patch_mask):
    mask = torch.ones([3,in_features.size(1), in_features.size(2)])
    img_mask = mask - patch_mask

    return img_mask

# add a patch to the original image
def add_patch(in_features, my_patch):
    
    # in_features: [1,3,416,416]
    patch_size = cfg_p.patch_size
    patch_mask = create_patch_mask(in_features, my_patch, patch_size)

    img_mask = create_img_mask(in_features, patch_mask)

    patch_mask = Variable(patch_mask.cuda(), requires_grad=False)
    img_mask = Variable(img_mask.cuda(), requires_grad=False)


    with_patch = in_features * img_mask + my_patch * patch_mask
    
    return with_patch

def save_patch(patch, epoch):
    patch_size = patch.size(2)
    patch_np = patch.data.cpu().numpy()
        
    save_patch_name = os.path.join(cfg_p.patch_dir, '{}.npy'.format(epoch))
    print("save patch as ", save_patch_name)
    np.save(save_patch_name, patch_np)

    patch_img_np = np.zeros((patch_size, patch_size,3))
    patch_img_np[:,:,0] = patch_np[0][0]*255.0 # B(0)
    patch_img_np[:,:,1] = patch_np[0][1]*255.0 # G(1)
    patch_img_np[:,:,2] = patch_np[0][2]*255.0 # R(2)
    np.transpose(patch_img_np, (2,1,0)) #RGB

    patch_img = Image.fromarray(patch_img_np.astype('uint8'))
    save_patch_img = os.path.join(cfg_p.patch_dir, '{}.png'.format(epoch))
    print("save patch as img ", save_patch_img)
    patch_img.save(save_patch_img)
