This directory contains the files I used to extract CNN features for my images. This is not meant to be a ready-to-use utility. I just wanted to provide this code in hope that it can be useful to someone as starter code to work on top of.

- This code uses [Caffe](http://caffe.berkeleyvision.org/) and their Matlab wrapper.
- I use VGG Net which can be found in the [Model Zoo ](https://github.com/BVLC/caffe/wiki/Model-Zoo) under the title *Models used by the VGG team in ILSVRC-2014*. I use the [16-layer version](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md).
- Note that I provide my _features deploy network def as well, which is exactly what you see on that page but I chopped off the softmax to get the 4096-D codes below.
