import argparse
from pathlib import Path
# import PIL.Image as Image
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


def s(fig=plt, ax, x, title):
    cax = ax.imshow(x, cmap='gray', aspect='equal')
    min = np.amin(x)
    max = np.amax(x)
    ax.set_title(f'{title} {min:.2f}~{max:.2f}')
    # fig.colorbar(cax)


def show(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# matplotlib.rcParams['text.usetex'] = True
img = imread('BSD/test007.png')
img = np.array(img, dtype=np.float32) / 255.0
# plt.figure(figsize = (2,2))
gs1 = gridspec.GridSpec(6, 2, wspace=0, hspace=0)
# gs1.update()  # set the spacing between axes.
# plt.axis('off')
for i in range(3):
   # i = i + 1 # grid spec indexes from 0
    ax1 = plt.subplot(gs1[i,0])
    s(plt, ax1, img, 'la')
    # ax1.imshow(img, cmap='gray')
    # ax1.set_title('$\partial x_1$')

    # ax1.set_xticklabels([])
    ax1.set_xticks([])
    # ax1.set_yticklabels([])
    ax1.set_yticks([])
    # ax1.set_title('ddd')
    # ax1.set_aspect('equal')
ax1 = plt.subplot(gs1[:5,:])
ax1.plot([1,2,3],[2,3,4])
ax1.set_xlabel('ss')
ax1.set_ylabel('ss')
# s(plt, ax1, img, 'la')
plt.show()

# gs =
