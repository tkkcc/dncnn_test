#!/bin/bash
echo 'model downloading'
wget 'https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth?raw=true' -O model.pth
echo 'model downloaded'
mkdir -p BSD
wget 'https://raw.githubusercontent.com/cszn/DnCNN/master/testsets/BSD68/test007.png' -O BSD/test007.png
echo 'finish'
# echo 'Pleas download BSD from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/'
