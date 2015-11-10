import sys
import os.path
import argparse

import numpy as np
from scipy.misc import imread, imresize
import scipy.io

import cPickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument('--caffe',
                    help='path to caffe installation')
parser.add_argument('--model_def',
                    help='path to model definition prototxt')
parser.add_argument('--model',
                    help='path to model parameters')
parser.add_argument('--files',
                    help='path to a file contsining a list of images')
parser.add_argument('--gpu',
                    action='store_true',
                    help='whether to use gpu training')
parser.add_argument('--out',
                    help='name of the pickle file where to store the features')

args = parser.parse_args()

if args.caffe:
    caffepath = args.caffe + '/python'
    sys.path.append(caffepath)

import caffe

def predict(in_data, net):
    """
    Get the features for a batch of data using network

    Inputs:
    in_data: data batch
    """

    out = net.forward(**{net.inputs[0]: in_data})
    features = out[net.outputs[0]]
    return features


def batch_predict(filenames, net):
    """
    Get the features for all images from filenames using a network

    Inputs:
    filenames: a list of names of image files

    Returns:
    an array of feature vectors for the images in that file
    """

    N, C, H, W = net.blobs[net.inputs[0]].data.shape
    F = net.blobs[net.outputs[0]].data.shape[1]
    Nf = len(filenames)
    Hi, Wi, _ = imread(filenames[0]).shape
    allftrs = np.zeros((Nf, F))
    for i in range(0, Nf, N):
        in_data = np.zeros((N, C, H, W), dtype=np.float32)

        batch_range = range(i, min(i+N, Nf))
        batch_filenames = [filenames[j] for j in batch_range]
        Nb = len(batch_range)

        batch_images = np.zeros((Nb, 3, H, W))
        for j,fname in enumerate(batch_filenames):
            im = imread(fname)
            if len(im.shape) == 2:
                im = np.tile(im[:,:,np.newaxis], (1,1,3))
            # RGB -> BGR
            im = im[:,:,(2,1,0)]
            # mean subtraction
            im = im - np.array([103.939, 116.779, 123.68])
            # resize
            im = imresize(im, (H, W), 'bicubic')
            # get channel in correct dimension
            im = np.transpose(im, (2, 0, 1))
            batch_images[j,:,:,:] = im

        # insert into correct place
        in_data[0:len(batch_range), :, :, :] = batch_images

        # predict features
        ftrs = predict(in_data, net)

        for j in range(len(batch_range)):
            allftrs[i+j,:] = ftrs[j,:]

        print 'Done %d/%d files' % (i+len(batch_range), len(filenames))

    return allftrs


if args.gpu:
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()

net = caffe.Net(args.model_def, args.model, caffe.TEST)

filenames = []

base_dir = os.path.dirname(args.files)
with open(args.files) as fp:
    for line in fp:
        filename = os.path.join(base_dir, line.strip().split()[0])
        filenames.append(filename)

allftrs = batch_predict(filenames, net)

if args.out:
    # store the features in a pickle file
    with open(args.out, 'w') as fp:
        pickle.dump(allftrs, fp)

scipy.io.savemat(os.path.join(base_dir, 'vgg_feats.mat'), mdict =  {'feats': np.transpose(allftrs)})
