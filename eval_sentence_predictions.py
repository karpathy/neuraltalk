import argparse
import json
import time
import datetime
import numpy as np
import code
import socket
import os
import cPickle as pickle
import math

from imagernn.data_provider import getDataProvider
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split

from nltk.align.bleu import BLEU

# UTILS needed for BLEU score evaluation      
def BLEUscore(candidate, references, weights):
  p_ns = [BLEU.modified_precision(candidate, references, i) for i, _ in enumerate(weights, start=1)]
  if all([x > 0 for x in p_ns]):
      s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns))
      bp = BLEU.brevity_penalty(candidate, references)
      return bp * math.exp(s)
  else: # this is bad
      return 0

def evalCandidate(candidate, references):
  """ 
  candidate is a single list of words, references is a list of lists of words
  written by humans.
  """
  b1 = BLEUscore(candidate, references, [1.0])
  b2 = BLEUscore(candidate, references, [0.5, 0.5])
  b3 = BLEUscore(candidate, references, [1/3.0, 1/3.0, 1/3.0])
  return [b1,b2,b3]

def main(params):

  # load the checkpoint
  checkpoint_path = params['checkpoint_path']
  max_images = params['max_images']

  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  checkpoint_params = checkpoint['params']
  dataset = checkpoint_params['dataset']
  model = checkpoint['model']

  # fetch the data provider
  dp = getDataProvider(dataset)

  misc = {}
  misc['wordtoix'] = checkpoint['wordtoix']
  ixtoword = checkpoint['ixtoword']

  blob = {} # output blob which we will dump to JSON for visualizing the results
  blob['params'] = params
  blob['checkpoint_params'] = checkpoint_params
  blob['imgblobs'] = []

  # iterate over all images in test set and predict sentences
  BatchGenerator = decodeGenerator(checkpoint_params)
  all_bleu_scores = []
  n = 0
  #for img in dp.iterImages(split = 'test', shuffle = True, max_images = max_images):
  for img in dp.iterImages(split = 'test', max_images = max_images):
    n+=1
    print 'image %d/%d:' % (n, max_images)
    references = [x['tokens'] for x in img['sentences']] # as list of lists of tokens
    kwparams = { 'beam_size' : params['beam_size'] }
    Ys = BatchGenerator.predict([{'image':img}], model, checkpoint_params, **kwparams)

    img_blob = {} # we will build this up
    img_blob['img_path'] = img['local_file_path']
    img_blob['imgid'] = img['imgid']

    # encode the human-provided references
    img_blob['references'] = []
    for gtwords in references:
      print 'GT: ' + ' '.join(gtwords)
      img_blob['references'].append({'text': ' '.join(gtwords)})

    # now evaluate and encode the top prediction
    top_predictions = Ys[0] # take predictions for the first (and only) image we passed in
    top_prediction = top_predictions[0] # these are sorted with highest on top
    candidate = [ixtoword[ix] for ix in top_prediction[1]]
    print 'PRED: (%f) %s' % (top_prediction[0], ' '.join(candidate))
    bleu_scores = evalCandidate(candidate, references)
    print 'BLEU: B-1: %f B-2: %f B-3: %f' % tuple(bleu_scores)
    img_blob['candidate'] = {'text': ' '.join(candidate), 'logprob': top_prediction[0], 'bleu': bleu_scores}

    all_bleu_scores.append(bleu_scores)
    blob['imgblobs'].append(img_blob)

  print 'final average bleu scores:'
  bleu_averages = [sum(x[i] for x in all_bleu_scores)*1.0/len(all_bleu_scores) for i in xrange(3)]
  blob['final_result'] = { 'bleu' : bleu_averages }
  print 'FINAL BLEU: B-1: %f B-2: %f B-3: %f' % tuple(bleu_averages)
  
  # now also evaluate test split perplexity
  gtppl = eval_split('test', dp, model, checkpoint_params, misc, eval_max_images = max_images)
  print 'perplexity of ground truth words: %f' % (gtppl, )
  blob['gtppl'] = gtppl

  # dump result struct to file
  print 'saving result struct to %s' % (params['result_struct_filename'], )
  json.dump(blob, open(params['result_struct_filename'], 'w'))

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_path', type=str, help='the input checkpoint')
  parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size in inference. 1 indicates greedy per-word max procedure. Good value is approx 20 or so, and more = better.')
  parser.add_argument('--result_struct_filename', type=str, default='result_struct.json', help='filename of the result struct to save')
  parser.add_argument('-m', '--max_images', type=int, default=-1, help='max images to use')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
