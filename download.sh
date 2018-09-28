#!/bin/sh
echo 'model downloading'
wget 'https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth?raw=true' -O model.pth
echo 'model downloaded'
echo 'Pleas download BSD from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/'
