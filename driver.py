import argparse
import json
import time
import datetime
import numpy as np
import code
import socket
import os
import sys
import cPickle as pickle

from imagernn.data_provider import getDataProvider
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split

def preProBuildWordVocab(sentence_iterator, word_count_threshold):
  # count up all word counts so that we can threshold
  # this shouldnt be too expensive of an operation
  print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
  t0 = time.time()
  word_counts = {}
  nsents = 0
  for sent in sentence_iterator:
    nsents += 1
    for w in sent['tokens']:
      word_counts[w] = word_counts.get(w, 0) + 1
  vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
  print 'filtered words from %d to %d in %.2fs' % (len(word_counts), len(vocab), time.time() - t0)

  # with K distinct words:
  # - there are K+1 possible inputs (START token and all the words)
  # - there are K+1 possible outputs (END token and all the words)
  # we use ixtoword to take predicted indeces and map them to words for output visualization
  # we use wordtoix to take raw words and get their index in word vector matrix
  ixtoword = {}
  ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
  wordtoix = {}
  wordtoix['#START#'] = 0 # make first vector be the start token
  ix = 1
  for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

  # compute bias vector, which is related to the log probability of the distribution
  # of the labels (words) and how often they occur. We will use this vector to initialize
  # the decoder weights, so that the loss function doesnt show a huge increase in performance
  # very quickly (which is just the network learning this anyway, for the most part). This makes
  # the visualizations of the cost function nicer because it doesn't look like a hockey stick.
  # for example on Flickr8K, doing this brings down initial perplexity from ~2500 to ~170.
  word_counts['.'] = nsents
  bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
  bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
  bias_init_vector = np.log(bias_init_vector)
  bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
  return wordtoix, ixtoword, bias_init_vector

def RNNGenCost(batch, model, params, misc):
  """ cost function, returns cost and gradients for model """
  regc = params['regc'] # regularization cost
  BatchGenerator = decodeGenerator(params)
  wordtoix = misc['wordtoix']

  # forward the RNN on each image sentence pair
  # the generator returns a list of matrices that have word probabilities
  # and a list of cache objects that will be needed for backprop
  Ys, gen_caches = BatchGenerator.forward(batch, model, params, misc, predict_mode = False)

  # compute softmax costs for all generated sentences, and the gradients on top
  loss_cost = 0.0
  dYs = []
  logppl = 0.0
  logppln = 0
  for i,pair in enumerate(batch):
    img = pair['image']
    # ground truth indeces for this sentence we expect to see
    gtix = [ wordtoix[w] for w in pair['sentence']['tokens'] if w in wordtoix ]
    gtix.append(0) # don't forget END token must be predicted in the end!
    # fetch the predicted probabilities, as rows
    Y = Ys[i]
    maxes = np.amax(Y, axis=1, keepdims=True)
    e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
    P = e / np.sum(e, axis=1, keepdims=True)
    loss_cost += - np.sum(np.log(1e-20 + P[range(len(gtix)),gtix])) # note: add smoothing to not get infs
    logppl += - np.sum(np.log2(1e-20 + P[range(len(gtix)),gtix])) # also accumulate log2 perplexities
    logppln += len(gtix)

    # lets be clever and optimize for speed here to derive the gradient in place quickly
    for iy,y in enumerate(gtix):
      P[iy,y] -= 1 # softmax derivatives are pretty simple
    dYs.append(P)

  # backprop the RNN
  grads = BatchGenerator.backward(dYs, gen_caches)

  # add L2 regularization cost and gradients
  reg_cost = 0.0
  if regc > 0:    
    for p in misc['regularize']:
      mat = model[p]
      reg_cost += 0.5 * regc * np.sum(mat * mat)
      grads[p] += regc * mat

  # normalize the cost and gradient by the batch size
  batch_size = len(batch)
  reg_cost /= batch_size
  loss_cost /= batch_size
  for k in grads: grads[k] /= batch_size

  # return output in json
  out = {}
  out['cost'] = {'reg_cost' : reg_cost, 'loss_cost' : loss_cost, 'total_cost' : loss_cost + reg_cost}
  out['grad'] = grads
  out['stats'] = { 'ppl2' : 2 ** (logppl / logppln)}
  return out

def main(params):
  batch_size = params['batch_size']
  dataset = params['dataset']
  word_count_threshold = params['word_count_threshold']
  do_grad_check = params['do_grad_check']
  max_epochs = params['max_epochs']
  host = socket.gethostname() # get computer hostname

  # fetch the data provider
  dp = getDataProvider(dataset)

  misc = {} # stores various misc items that need to be passed around the framework

  # go over all training sentences and find the vocabulary we want to use, i.e. the words that occur
  # at least word_count_threshold number of times
  misc['wordtoix'], misc['ixtoword'], bias_init_vector = preProBuildWordVocab(dp.iterSentences('train'), word_count_threshold)

  # delegate the initialization of the model to the Generator class
  BatchGenerator = decodeGenerator(params)
  init_struct = BatchGenerator.init(params, misc)
  model, misc['update'], misc['regularize'] = (init_struct['model'], init_struct['update'], init_struct['regularize'])

  # force overwrite here. This is a bit of a hack, not happy about it
  model['bd'] = bias_init_vector.reshape(1, bias_init_vector.size)

  print 'model init done.'
  print 'model has keys: ' + ', '.join(model.keys())
  print 'updating: ' + ', '.join( '%s [%dx%d]' % (k, model[k].shape[0], model[k].shape[1]) for k in misc['update'])
  print 'updating: ' + ', '.join( '%s [%dx%d]' % (k, model[k].shape[0], model[k].shape[1]) for k in misc['regularize'])
  print 'number of learnable parameters total: %d' % (sum(model[k].shape[0] * model[k].shape[1] for k in misc['update']), )

  if params.get('init_model_from', ''):
    # load checkpoint
    checkpoint = pickle.load(open(params['init_model_from'], 'rb'))
    model = checkpoint['model'] # overwrite the model

  # initialize the Solver and the cost function
  solver = Solver()
  def costfun(batch, model):
    # wrap the cost function to abstract some things away from the Solver
    return RNNGenCost(batch, model, params, misc)

  # calculate how many iterations we need
  num_sentences_total = dp.getSplitSize('train', ofwhat = 'sentences')
  num_iters_one_epoch = num_sentences_total / batch_size
  max_iters = max_epochs * num_iters_one_epoch
  eval_period_in_epochs = params['eval_period']
  eval_period_in_iters = max(1, int(num_iters_one_epoch * eval_period_in_epochs))
  abort = False
  top_val_ppl2 = -1
  smooth_train_ppl2 = len(misc['ixtoword']) # initially size of dictionary of confusion
  val_ppl2 = len(misc['ixtoword'])
  last_status_write_time = 0 # for writing worker job status reports
  json_worker_status = {}
  json_worker_status['params'] = params
  json_worker_status['history'] = []
  for it in xrange(max_iters):
    if abort: break
    t0 = time.time()
    # fetch a batch of data
    batch = [dp.sampleImageSentencePair() for i in xrange(batch_size)]
    # evaluate cost, gradient and perform parameter update
    step_struct = solver.step(batch, model, costfun, **params)
    cost = step_struct['cost']
    dt = time.time() - t0

    # print training statistics
    train_ppl2 = step_struct['stats']['ppl2']
    smooth_train_ppl2 = 0.99 * smooth_train_ppl2 + 0.01 * train_ppl2 # smooth exponentially decaying moving average
    if it == 0: smooth_train_ppl2 = train_ppl2 # start out where we start out
    epoch = it * 1.0 / num_iters_one_epoch
    print '%d/%d batch done in %.3fs. at epoch %.2f. loss cost = %f, reg cost = %f, ppl2 = %.2f (smooth %.2f)' \
          % (it, max_iters, dt, epoch, cost['loss_cost'], cost['reg_cost'], \
             train_ppl2, smooth_train_ppl2)

    # perform gradient check if desired, with a bit of a burnin time (10 iterations)
    if it == 10 and do_grad_check:
      print 'disabling dropout for gradient check...'
      params['drop_prob_encoder'] = 0
      params['drop_prob_decoder'] = 0
      solver.gradCheck(batch, model, costfun)
      print 'done gradcheck, exitting.'
      sys.exit() # hmmm. probably should exit here

    # detect if loss is exploding and kill the job if so
    total_cost = cost['total_cost']
    if it == 0:
      total_cost0 = total_cost # store this initial cost
    if total_cost > total_cost0 * 2:
      print 'Aboring, cost seems to be exploding. Run gradcheck? Lower the learning rate?'
      abort = True # set the abort flag, we'll break out

    # logging: write JSON files for visual inspection of the training
    tnow = time.time()
    if tnow > last_status_write_time + 60*1: # every now and then lets write a report
      last_status_write_time = tnow
      jstatus = {}
      jstatus['time'] = datetime.datetime.now().isoformat()
      jstatus['iter'] = (it, max_iters)
      jstatus['epoch'] = (epoch, max_epochs)
      jstatus['time_per_batch'] = dt
      jstatus['smooth_train_ppl2'] = smooth_train_ppl2
      jstatus['val_ppl2'] = val_ppl2 # just write the last available one
      jstatus['train_ppl2'] = train_ppl2
      json_worker_status['history'].append(jstatus)
      status_file = os.path.join(params['worker_status_output_directory'], host + '_status.json')
      try:
        json.dump(json_worker_status, open(status_file, 'w'))
      except Exception, e: # todo be more clever here
        print 'tried to write worker status into %s but got error:' % (status_file, )
        print e

    # perform perplexity evaluation on the validation set and save a model checkpoint if it's good
    is_last_iter = (it+1) == max_iters
    if (((it+1) % eval_period_in_iters) == 0 and it < max_iters - 5) or is_last_iter:
      val_ppl2 = eval_split('val', dp, model, params, misc) # perform the evaluation on VAL set
      print 'validation perplexity = %f' % (val_ppl2, )
      
      # abort training if the perplexity is no good
      min_ppl_or_abort = params['min_ppl_or_abort']
      if val_ppl2 > min_ppl_or_abort and min_ppl_or_abort > 0:
        print 'aborting job because validation perplexity %f < %f' % (val_ppl2, min_ppl_or_abort)
        abort = True # abort the job

      write_checkpoint_ppl_threshold = params['write_checkpoint_ppl_threshold']
      if val_ppl2 < top_val_ppl2 or top_val_ppl2 < 0:
        if val_ppl2 < write_checkpoint_ppl_threshold or write_checkpoint_ppl_threshold < 0:
          # if we beat a previous record or if this is the first time
          # AND we also beat the user-defined threshold or it doesnt exist
          top_val_ppl2 = val_ppl2
          filename = 'model_checkpoint_%s_%s_%s_%.2f.p' % (dataset, host, params['fappend'], val_ppl2)
          filepath = os.path.join(params['checkpoint_output_directory'], filename)
          checkpoint = {}
          checkpoint['it'] = it
          checkpoint['epoch'] = epoch
          checkpoint['model'] = model
          checkpoint['params'] = params
          checkpoint['perplexity'] = val_ppl2
          checkpoint['wordtoix'] = misc['wordtoix']
          checkpoint['ixtoword'] = misc['ixtoword']
          try:
            pickle.dump(checkpoint, open(filepath, "wb"))
            print 'saved checkpoint in %s' % (filepath, )
          except Exception, e: # todo be more clever here
            print 'tried to write checkpoint into %s but got error: ' % (filepat, )
            print e


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument('-d', '--dataset', dest='dataset', default='flickr8k', help='dataset: flickr8k/flickr30k')
  parser.add_argument('-a', '--do_grad_check', dest='do_grad_check', type=int, default=0, help='perform gradcheck? program will block for visual inspection and will need manual user input')
  parser.add_argument('--fappend', dest='fappend', type=str, default='baseline', help='append this string to checkpoint filenames')
  parser.add_argument('-o', '--checkpoint_output_directory', dest='checkpoint_output_directory', type=str, default='cv/', help='output directory to write checkpoints to')
  parser.add_argument('--worker_status_output_directory', dest='worker_status_output_directory', type=str, default='status/', help='directory to write worker status JSON blobs to')
  parser.add_argument('--write_checkpoint_ppl_threshold', dest='write_checkpoint_ppl_threshold', type=float, default=-1, help='ppl threshold above which we dont bother writing a checkpoint to save space')
  parser.add_argument('--init_model_from', dest='init_model_from', type=str, default='', help='initialize the model parameters from some specific checkpoint?')
  
  # model parameters
  parser.add_argument('--generator', dest='generator', type=str, default='lstm', help='generator to use')
  parser.add_argument('--image_encoding_size', dest='image_encoding_size', type=int, default=256, help='size of the image encoding')
  parser.add_argument('--word_encoding_size', dest='word_encoding_size', type=int, default=256, help='size of word encoding')
  parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=256, help='size of hidden layer in generator RNNs')
  # lstm-specific params
  parser.add_argument('--tanhC_version', dest='tanhC_version', type=int, default=0, help='use tanh version of LSTM?')
  # rnn-specific params
  parser.add_argument('--rnn_relu_encoders', dest='rnn_relu_encoders', type=int, default=0, help='relu encoders before going to RNN?')
  parser.add_argument('--rnn_feed_once', dest='rnn_feed_once', type=int, default=0, help='feed image to the rnn only single time?')

  # optimization parameters
  parser.add_argument('-c', '--regc', dest='regc', type=float, default=1e-8, help='regularization strength')
  parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=50, help='number of epochs to train for')
  parser.add_argument('--solver', dest='solver', type=str, default='rmsprop', help='solver type: vanilla/adagrad/adadelta/rmsprop')
  parser.add_argument('--momentum', dest='momentum', type=float, default=0.0, help='momentum for vanilla sgd')
  parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.999, help='decay rate for adadelta/rmsprop')
  parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, default=1e-8, help='epsilon smoothing for rmsprop/adagrad/adadelta')
  parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-3, help='solver learning rate')
  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=100, help='batch size')
  parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=5, help='clip gradients (normalized by batch size)? elementwise. if positive, at what threshold?')
  parser.add_argument('--drop_prob_encoder', dest='drop_prob_encoder', type=float, default=0.5, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--drop_prob_decoder', dest='drop_prob_decoder', type=float, default=0.5, help='what dropout to apply right before the decoder in an RNN/LSTM')

  # data preprocessing parameters
  parser.add_argument('--word_count_threshold', dest='word_count_threshold', type=int, default=5, help='if a word occurs less than this number of times in training data, it is discarded')

  # evaluation parameters
  parser.add_argument('-p', '--eval_period', dest='eval_period', type=float, default=1.0, help='in units of epochs, how often do we evaluate on val set?')
  parser.add_argument('--eval_batch_size', dest='eval_batch_size', type=int, default=100, help='for faster validation performance evaluation, what batch size to use on val img/sentences?')
  parser.add_argument('--eval_max_images', dest='eval_max_images', type=int, default=-1, help='for efficiency we can use a smaller number of images to get validation error')
  parser.add_argument('--min_ppl_or_abort', dest='min_ppl_or_abort', type=float , default=-1, help='if validation perplexity is below this threshold the job will abort')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)