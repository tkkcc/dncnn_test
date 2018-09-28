import argparse
from pathlib import Path
# import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='.', type=str,
                        help='directory of test dataset')
    parser.add_argument(
        '--set_names', default=['BSD'], help='directory of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default='.',
                        help='directory of the model')
    parser.add_argument('--model_name', default='model.pth',
                        type=str, help='the model name')
    parser.add_argument('--result_dir', default='results',
                        type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=0, type=int,
                        help='save the denoised image, 1 or 0')
    return parser.parse_args()


def show(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                                kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(
                n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels,
                                kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return out
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


def n(x,x_):
    return x.view(*x_.shape).detach().cpu().numpy().astype(np.float32)

def s(fig,ax,x,title):
    im = ax.imshow(x, cmap='gray')
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

if __name__ == '__main__':
    args = parse_args()
    model = torch.load(Path(args.model_dir, 'model.pth'), map_location='cpu')
    model.eval()
    cuda = torch.cuda.is_available()
    model = model.cuda() if cuda else model
    img = imread('BSD/test007.png')
    # img = resize(img, (100, 100))
    fig, ax = plt.subplots(2, 3)
    x_ = np.array(img, dtype=np.float32) / 255.0
    s(fig,ax[0,0],x_,'original')
    # im = ax[0, 0].imshow(x_, cmap='gray')
    
    # ax[0, 0].set_title('original')

    tmp = torch.from_numpy(x_).view(1, -1, *x_.shape)
    tmp = tmp.cuda() if cuda else tmp
    np.random.seed(seed=0)  # for reproducibility
    noise = np.random.normal(0, args.sigma / 255.0, x_.shape)
    # ax[0, 1].imshow(noise, cmap='gray')
    s(fig,ax[0,1],noise,'noise')

    y_ = x_ + noise
    # ax[0, 2].imshow(y_, cmap='gray')
    # ax[0, 2].set_title('original+noise')
    y_ = y_.astype(np.float32)
    s(fig,ax[0,2],y_,'original+noise=input')
    y = torch.from_numpy(y_).view(1, -1, *y_.shape)
    y = y.cuda() if cuda else y
    # dncnn result
    # t = model(y).view(*x_.shape).cpu().detach().numpy().astype(np.float32)
    # psnr = compare_psnr(x_, t)
    # show(np.hstack((x_, t)))
    # print('dncnn psnr=%2.2fdb' % psnr)
    # iterate result
    x = torch.tensor(y, requires_grad=True)
    x = x.cuda() if cuda else x
    lr = 0.01
    mse = nn.MSELoss(reduction='sum')
    l1 = nn.L1Loss(reduction='sum')
    iter_num = 30
    psnrs = []
    psnrms = []
    t = {}
    model.zero_grad()
    model_out = model(x)
    s(fig,ax[1,0],n(model_out),'output')
    # ax[1, 0].imshow(n(model_out), cmap='gray')
    # ax[1, 0].set_title('output')
    model_loss = model_out.sum()
    model_loss.backward()
    s(fig,ax[1,1],n(x-model_out),'input-output')
    s(fig,ax[1,2],n(x.grad),'gradient of input')
    # ax[1, 1].imshow(n(x.grad), cmap='gray')
    # ax[1, 1].set_title('gradient of input')
    ax[1,2].axis('off')
    # fig.colorbar(im, ax=ax[0, 0])
    # fig.colorbar(im, ax=ax[0, 1])
    # fig.colorbar(im, ax=ax[0, 2])
    # fig.colorbar(im, ax=ax[1, 0])
    # fig.colorbar(im, ax=ax[1, 1])
    for name, parameter in model.named_parameters():
        t[name] = parameter.grad
    fig,ax=plt.subplots(1, 2)
    s(fig,ax[0,0],n(model_out),'output')
    s(fig,ax[0,1],n(0.01*x.grad),'0.01 * gradient of input')
    fig.show()
    pass
