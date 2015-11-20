#NeuralTalk

**Warning: Deprecated.**
Hi there, this code is now quite old and inefficient, and now deprecated. I am leaving it on Github for educational purposes, but if you would like to run or train image captioning I warmly recommend my new code release [NeuralTalk2](https://github.com/karpathy/neuraltalk2). NeuralTalk2 is written in [Torch](http://torch.ch/) and is SIGNIFICANTLY (I mean, ~100x+) faster because it is batched and runs on the GPU. It also supports CNN finetuning, which helps a lot with performance.


This project contains *Python+numpy* source code for learning **Multimodal Recurrent Neural Networks** that describe images with sentences.

This line of work was recently featured in a [New York Times article](http://www.nytimes.com/2014/11/18/science/researchers-announce-breakthrough-in-content-recognition-software.html) and has been the subject of multiple academic papers from the research community over the last few months. This code currently implements the models proposed by [Vinyals et al. from Google (CNN + LSTM)](http://arxiv.org/abs/1411.4555) and by [Karpathy and Fei-Fei from Stanford (CNN + RNN)](http://cs.stanford.edu/people/karpathy/deepimagesent/). Both models take an image and predict its sentence description with a Recurrent Neural Network (either an LSTM or an RNN).

## Overview
The pipeline for the project looks as follows:

- The **input** is a dataset of images and 5 sentence descriptions that were collected with Amazon Mechanical Turk. In particular, this code base is set up for [Flickr8K](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/), and [MSCOCO](http://mscoco.org/) datasets. 
- In the **training stage**, the images are fed as input to RNN and the RNN is asked to predict the words of the sentence, conditioned on the current word and previous context as mediated by the hidden layers of the neural network. In this stage, the parameters of the networks are trained with backpropagation.
- In the **prediction stage**, a witheld set of images is passed to RNN and the RNN generates the sentence one word at a time. The results are evaluated with **BLEU score**. The code also includes utilities for visualizing the results in HTML.

## Dependencies
**Python 2.7**, modern version of **numpy/scipy**, **perl** (if you want to do BLEU score evaluation), **argparse** module. Most of these are okay to install with **pip**. To install all dependencies at once, run the command `pip install -r requirements.txt`

I only tested this code with Ubuntu 12.04, but I tried to make it as generic as possible (e.g. use of **os** module for file system interactions etc. So it might work on Windows and Mac relatively easily.)

*Protip*: you really want to link your numpy to use a BLAS implementation for its matrix operations. I use **virtualenv** and link numpy against a system installation of **OpenBLAS**. Doing this will make this code almost an order of time faster because it relies very heavily on large matrix multiplies.

## Getting started

1. **Get the code.** `$ git clone` the repo and install the Python dependencies
2. **Get the data.** I don't distribute the data in the Git repo, instead download the `data/` folder from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/). Also, this download does not include the raw image files, so if you want to visualize the annotations on raw images, you have to obtain the images from Flickr8K / Flickr30K / COCO directly and dump them into the appropriate data folder.
3. **Train the model.** Run the training `$ python driver.py` (see many additional argument settings inside the file) and wait. You'll see that the learning code writes checkpoints into `cv/` and periodically reports its status in `status/` folder. 
4. **Monitor the training.** The status can be inspected manually by reading the JSON and printing whatever you wish in a second process. In practice I run cross-validations on a cluster, so my `cv/` folder fills up with a lot of checkpoints that I further filter and inspect with other scripts. I am including my cluster training status visualization utility as well if you like. Run a local webserver (e.g. `$ python -m SimpleHTTPServer 8123`) and then open `monitorcv.html` in your browser on `http://localhost:8123/monitorcv.html`, or whatever the web server tells you the path is. You will have to edit the file to setup the paths properly and point it at the right json files.
5. **Evaluate model checkpoints.** To evaluate a checkpoint from `cv/`, run the `evaluate_sentence_predctions.py` script and pass it the path to a checkpoint.
6. **Visualize the predictions.** Use the included html file `visualize_result_struct.html` to visualize the JSON struct produced by the evaluation code. This will visualize the images and their predictions. Note that you'll have to download the raw images from the individual dataset pages and place them into the corresponding `data/` folder.

Lastly, note that this is currently research code, so a lot of the documentation is inside individual Python files. If you wish to work with this code, you'll have to get familiar with it and be comfortable reading Python code.

## Pretrained model

Some pretrained models can be found in the [NeuralTalk Model Zoo](http://cs.stanford.edu/people/karpathy/neuraltalk/). The slightly hairy part is that if you wish to apply these models to some arbitrary new image (one not from Flickr8k/30k/COCO) you have to first extract the CNN features. I use the 16-layer [VGG network](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) from Simonyan and Zisserman, because the model is beautiful, powerful and available with [Caffe](http://caffe.berkeleyvision.org/). There is opportunity for putting the preprocessing and inference into a single nice function that uses the Python wrapper to get the features and then runs the pretrained sentence model. I might add this in the future.

## Using the model to predict on new images

The code allows you to easily predict and visualize results of running the model on COCO/Flickr8K/Flick30K images. If you want to run the code on arbitrary image (e.g. on your file system), things get a little more complicated because we need to first need to pipe your image through the VGG CNN to get the 4096-D activations on top. 

Have a look inside the folder `example_images` for instructions on how to do this. Currently, the code for extracting the raw features from each image is in Matlab, so you will need it installed on your system. Caffe also has a wrapper for Python, but I wasn't yet able to use the Python wrapper to exactly reproduce the features I get from Matlab. The `example_images` will walk you through the process, and you will eventually use `predict_on_images.py` to run the prediction.

## Using your own data

The input to the system is the **data** folder, which contains the Flickr8K, Flickr30K and MSCOCO datasets. In particular, each folder (e.g. `data/flickr8k`) contains a `dataset.json` file that stores the image paths and sentences in the dataset (all images, sentences, raw preprocessed tokens, splits, and the mappings between images and sentences). Each folder additionally contains `vgg_feats.mat` , which is a `.mat` file that stores the CNN features from all images, one per row, using the VGG Net from ILSVRC 2014. Finally, there is the `imgs/` folder that holds the raw images. I also provide the Matlab script that I used to extract the features, which you may find helpful if you wish to use a different dataset. This is inside the `matlab_features_reference/` folder, and see the Readme file in that folder for more information.

## License
BSD license.
