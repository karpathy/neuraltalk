This directory contains a Python port of the Matlab code in matlab_features_reference/ directory

- This code uses [Caffe](http://caffe.berkeleyvision.org/) and their Python wrapper.
- I use VGG Net which can be found in the [Model Zoo ](https://github.com/BVLC/caffe/wiki/Model-Zoo) under the title *Models used by the VGG team in ILSVRC-2014*. I use the [16-layer version](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md).
- Note that I provide my _features deploy network def as well, which is exactly what you see on that page but I chopped off the softmax to get the 4096-D codes below.
