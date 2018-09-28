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


def n(x, x_):
    return x.view(*x_.shape).detach().cpu().numpy().astype(np.float32)


def s(fig, ax, x, title):
    im = ax.imshow(x, cmap='gray')
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


if __name__ == '__main__':
    args = parse_args()
    cuda = torch.cuda.is_available()
    matplotlib.rc('font', size=18)
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    model = torch.load(Path(args.model_dir, 'model.pth'),
                       map_location='cuda:0')
    model.eval()
    model = model.cuda() if cuda else model
    img = imread('BSD/test007.png')
    img = resize(img, (100, 100))
    x_ = np.array(img, dtype=np.float32) / 255.0
    tmp = torch.from_numpy(x_).view(1, -1, *x_.shape)
    tmp = tmp.cuda() if cuda else tmp
    np.random.seed(seed=0)  # for reproducibility
    noise = np.random.normal(0, args.sigma/255.0, x_.shape)
    y_ = (x_ + noise).astype(np.float32)
    y = torch.from_numpy(y_).view(1, -1, *y_.shape)
    y = y.cuda() if cuda else y
    x = torch.tensor(y, requires_grad=True)
    x = x.cuda() if cuda else x
    lr = 0.01
    mse = nn.MSELoss(reduction='sum')
    l1 = nn.L1Loss(reduction='sum')
    iter_num = 2
    # x_k, x_k-y, dD_k/dx_k, dL(x_k)/dx_k, x_k+1
    fig, ax = plt.subplots(5, 30, figsize=(5, 30))
    psnrs = []
    psnrms = []
    t = {}
    
    for i in range(iter_num):
        
        model.zero_grad()
        model_out = model(x)
        model_loss = model_out.sum()
        model_loss.backward()
        # model_out.backward(torch.ones_like(model_out))
        # for name, parameter in model.named_parameters():
        #     t[name] = parameter.grad
        with torch.no_grad():
            s(fig,ax[i+1,0],n(x,x_),f'$x_{i}$')
            s(fig,ax[i+1,1],n(x-y,x_),f'$x_{i}-y$')
            g = (25/args.sigma)**2*(x-y)+x.grad
            s(fig,ax[i+1,2],n(x.grad,x_),f'$\partial D(x_{i})/\partial x_{i}$')
            s(fig,ax[i+1,2],n(g,x_),f'$\partial L(x_{i})/\partial x_{i}$')
            x -= lr * g
            x.grad.zero_()
            t = n(x,x_)
            psnr = compare_psnr(x_, t)
            ssim = compare_ssim(x_, t)
            s(fig,ax[i+1,2],n(x,x_),f'$x_{i+1}$ psnr={psnr:2.2f}db ssim={ssim:1.4f}')
            print(f'{i} lr={lr:.3f} psnr={psnr:2.2f}db')
            psnrs.append(psnr)
    # top summary plot
    ax0 = plt.subplot2grid((30, 5), (0, 0), colspan=5,rowspan=5)
    ax0.plot([i+1 for i in range(iter_num)], psnrs, label='$x_i$')
    ax0.plot([i+1 for i in range(iter_num)], [compare_psnr(x_,n(y-model(y),x_))]*iter_num, label='DnCNN')
    # ax0.xlabel('iteration')
    # ax0.ylabel('psnr')
    # ax0.legend()
    plt.show()

# if __name__ == '__main__':
#     args = parse_args()
#     model = torch.load(Path(args.model_dir, 'model.pth'), map_location='cpu')
#     model.eval()
#     cuda = torch.cuda.is_available()
#     model = model.cuda() if cuda else model
#     Path(args.result_dir).mkdir(parents=True, exist_ok=True)
#     for set_current in args.set_names:
#         Path(args.result_dir,set_current).mkdir(parents=True, exist_ok=True)
#         psnrs = []
#         ssims = []
#         for img in Path(args.set_dir, set_current).iterdir():
#             x = np.array(imread(img), dtype=np.float32)/255.0
#             np.random.seed(seed=0) # for reproducibility
#             y = x + np.random.normal(0, args.sigma/255.0, x.shape)
#             y = y.astype(np.float32)
#             y_ = torch.from_numpy(y).view(1, -1, *y.shape)
#             y_=y_.cuda() if cuda else y_
#             x_ = model(y_).view(*y.shape).cpu().detach().numpy().astype(np.float32)
#             # show(x_)
#             psnr = compare_psnr(x, x_)
#             ssim = compare_ssim(x, x_)
#             if args.save_result:
#                 name, ext = img.name, img.suffix
#                 show(np.hstack((y, x_)))
#             psnrs.append(psnr)
#             ssims.append(ssim)
#             print('%10s %10s PSNR=%2.2fdB' % (set_current, img, psnr))
#         psnr_avg = np.mean(psnrs)
#         ssim_avg = np.mean(ssims)
#         psnrs.append(psnr_avg)
#         ssims.append(ssim_avg)
#         print('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_current, psnr_avg, ssim_avg))