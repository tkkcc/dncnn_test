#!/bin/bash
echo 'model downloading'
wget 'https://raw.githubusercontent.com/cszn/DnCNN/master/TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth'
echo 'model downloaded'
mkdir -p BSD
wget 'https://raw.githubusercontent.com/cszn/DnCNN/master/testsets/BSD68/test007.png' -O BSD/test007.png
mkdir -p Set12
wget 'https://raw.githubusercontent.com/cszn/DnCNN/master/testsets/Set12/01.png' -O Set12/01.png
#https://github.com/uschmidt83/fourier-deconvolution-network
#wget 'https://raw.githubusercontent.com/uschmidt83/fourier-deconvolution-network/master/data/example1.dlm'
echo 'finish'
# echo 'Pleas download BSD from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/'
