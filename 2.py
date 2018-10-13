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
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
# import csv
import torch.nn.functional as F


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
            layers.append(nn.ReLU(inplacse=True))
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


def n(x, x_=None):
    return x.detach().squeeze().cpu().numpy().astype(np.float32)
    return x.view(*x_.shape).detach().cpu().numpy().astype(np.float32)


def s(fig, ax, x, title):
    min = np.amin(x)
    max = np.amax(x)
    # x=x.clip(0,1)
    cax = ax.imshow(x, cmap='gray', aspect='equal')
    # ax.set_title(f'{title} {min:.2f}~{max:.2f}', loc='left')
    ax.set_title(f'{title}', loc='left')
    ax.set_xticks([])
    ax.set_yticks([])

def pad(img, kernel,mode='replicate'):
    p = [(i-1)//2 for i in kernel.shape]
    return F.pad(img, (p[-1], p[-1], p[-2], p[-2]), mode)

def m():
    # torch.set_num_threads(2)
    args = parse_args()
    cuda = torch.cuda.is_available()
    model = torch.load(Path(args.model_dir, 'model.pth'),
                       map_location='cpu')
                       # map_location='cuda:0')
    model = model.cuda() if cuda else model
    model.eval()
    print(f'cuda: {cuda}')
    print(f'{torch.cuda.current_device()}')
    print(f'{torch.cuda.current_blas_handle()}')
    print(f'{torch.cuda.device_count()}')
    lr = 10
    iter_num = 500
    lam = 1
    img = imread('Set12/01.png')
    k_ = np.loadtxt('./example1.dlm').astype(np.float32)
    x_ = np.array(img, dtype=np.float32) / 255.0
    k = torch.from_numpy(k_).view(1, 1, *k_.shape)
    k = k.cuda() if cuda else k
    gt = torch.from_numpy(x_).view(1, -1, *x_.shape)
    gt = gt.cuda() if cuda else gt
    torch.manual_seed(0)
    y = pad(gt,k)
    y = F.conv2d(y, k)
    y += torch.randn_like(y) * args.sigma / 255
    x = torch.tensor(y, requires_grad=True)
    psnrs = []

    # plt.figure(figsize=(10,30))
    gs = gridspec.GridSpec(3, 2, wspace=0, hspace=0)
    pre = None
    t=4
    for i in range(iter_num):
        t*=0.9
        # lr = 6 + t
        lr = 1
        model.zero_grad()
        model_out = model(x)
        model_loss = lam * model_out.norm(2)+0.5*(F.conv2d(pad(x,k), k)-y).norm(2)
        model_loss.backward()
        with torch.no_grad():
            x -= lr * x.grad
            x.grad.zero_()
            psnr = compare_psnr(x_,n(x))
            print(f'{i} lr={lr:.3f} psnr={psnr:2.2f}')
            psnrs.append(psnr)
            if len(psnrs) > 5 and psnrs[-2] > psnrs[-1]:
                x=pre
                del psnrs[-1]
                break
            pre = x.clone()
    dncnn_out = n(y - model(y))
    dncnn_psnr = compare_psnr(x_, dncnn_out)
    start_psnr = compare_psnr(x_, n(y))
    return
    s(None, plt.subplot(gs[0,0]), x_, f'x = ground truth')
    s(None, plt.subplot(gs[0,1]), n(y), f'y = x+blur+noise, psnr={start_psnr:.2f}')
    s(None, plt.subplot(gs[1,0]), dncnn_out, f'dncnn(y), psnr={dncnn_psnr:.2f}')
    s(None, plt.subplot(gs[1,1]), n(x), f'iteration result, psnr={psnrs[-1]:.2f}')
    ax = plt.subplot(gs[2:,:])
    ax.plot([i+1 for i in range(len(psnrs))], psnrs, label='$x_i$')
    ax.set_xlabel('iteration')
    ax.set_ylabel('psnr')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.set_xticks([i + 1 for i in range(iter_num)])
    # ax.set_yticks(np.arange(20, 40, 0.2))
    ax.set_title(
        f'$\eta={lr:.2f},\lambda={lam}$, iteration num={len(psnrs)}, dncnn psnr={dncnn_psnr:.2f}')
    # ax.legend()
    # plt.tight_layout()
    plt.show()

m()
