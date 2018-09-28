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


if __name__ == '__main__':
    args = parse_args()
    model = torch.load(Path(args.model_dir, 'model.pth'), map_location='cpu')
    model.eval()
    cuda = torch.cuda.is_available()
    model = model.cuda() if cuda else model
    img = imread('BSD/test006.png')
    # img = resize(img, (100, 100))
    x_ = np.array(img, dtype=np.float32) / 255.0
    tmp = torch.from_numpy(x_).view(1, -1, *x_.shape)
    tmp = tmp.cuda() if cuda else tmp
    np.random.seed(seed=0)  # for reproducibility
    y_ = x_ + np.random.normal(0, args.sigma/255.0, x_.shape)
    y_ = y_.astype(np.float32)
    y = torch.from_numpy(y_).view(1, -1, *y_.shape)
    y = y.cuda() if cuda else y
    # dncnn result
    t = model(y).view(*x_.shape).cpu().detach().numpy().astype(np.float32)
    psnr = compare_psnr(x_, t)
    show(np.hstack((x_, t)))
    print('dncnn psnr=%2.2fdb' % psnr)
    # iterate result
    x = torch.tensor(y, requires_grad=True)
    x = x.cuda() if cuda else x
    lr = 0.01
    mse = nn.MSELoss(reduction='sum')
    l1 = nn.L1Loss(reduction='sum')
    iter_num = 30
    psnrs=[]
    psnrms=[]
    for i in range(iter_num):
        if i == 20:
            show(x.view(*x_.shape).cpu().detach().numpy().astype(np.float32))
            lr /= 10
        # elif i == 60:
        #     lr /=102
        # show(x.view(*x_.shape).cpu().detach().numpy().astype(np.float32))
        model.zero_grad()
        m = model(x)
        # loss = (x - y).pow(2).sum() / (2 * args.sigma / 25) + torch.abs(x -  model(x)).sum()
        # loss = mse(x, y) / (2 * args.sigma / 25) + torch.sum(x - model(x))
        # loss = mse(x, y) / (2 * args.sigma / 25) + l1(x, m)
        loss = mse(x, y)/2 + l1(x, m)
        # print(mse(tmp,model(x)),mse(tmp,x))
        # loss = mse(x, y)/2 + l1(x, tmp)
        loss.backward()
        with torch.no_grad():
            x -= lr * x.grad
            x.grad.zero_()
            t = x.view(*x_.shape).cpu().detach().numpy().astype(np.float32)
            psnr = compare_psnr(x_, t)
            # show(t)
            t = m.view(*x_.shape).cpu().detach().numpy().astype(np.float32)
            psnrm = compare_psnr(x_, t)
            # show(np.hstack((x_, t)))
            print('%d lr=%.3f psnr=%2.2fdb psnrm=%2.2fdb loss=%2.2f' %
                  (i, lr, psnr, psnrm, loss))
            psnrs.append(psnr)
            psnrms.append(psnrm)
    matplotlib.rc('font', size=20)
    plt.plot([i+1 for i in range(iter_num)], psnrs, label='x')
    plt.plot([i+1 for i in range(iter_num)], psnrms, label='D(x)')
    plt.xlabel('iteration')
    plt.ylabel('psnr')
    plt.legend()
    plt.show()

            # print(psnr)
    # print(y_.cpu().numpy()-x_)
    # show(x_)
    # show(np.hstack((y, x_)))
    # print('PSNR=%2.2fdB' % psnr)

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
