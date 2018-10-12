import torch.nn.functional as F
import torch
import numpy as np
def pad_for_kernel(img,kernel,mode='edge'):
    p = [(d-1)//2 for d in kernel.shape]
    padding = [p,p]
    return np.pad(img, padding, mode)
filters = np.random.randn( 5, 5)
img = np.array()
a = pad_for_kernel(img, filters)
# out = F.conv2d(inputs, filters, padding=0)
...
    # y  = to_tensor(edgetaper(pad_for_kernel(img,kernel,'edge'),kernel))
