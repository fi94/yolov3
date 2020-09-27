import os
import torch
from torch.autograd import Variable
import numpy as np
# properties of DPatch
target_class = 0
patch_x, patch_y = 0., 0.
patch_w, patch_h = 120., 120.
patch_size = int(patch_w-patch_x)
patch_dir = os.path.join('trained_patch', str(target_class))
img_w, img_h = 416, 416
if not os.path.exists(patch_dir):
     os.mkdir(patch_dir)

patch_path = './trained_patch/0/9.npy'
patch = Variable(torch.FloatTensor(np.load(patch_path)),requires_grad=False)