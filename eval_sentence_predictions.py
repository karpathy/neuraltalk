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

def main(params):

  # load the checkpoint
  checkpoint_path = params['checkpoint_path']
  max_images = params['max_images']

  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  checkpoint_params = checkpoint['params']
  dataset = checkpoint_params['dataset']
  model = checkpoint['model']
  dump_folder = params['dump_folder']

  if dump_folder:
    print 'creating dump folder ' + dump_folder
    os.system('mkdir -p ' + dump_folder)
    
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
  n = 0
  all_references = []
  all_candidates = []
  for img in dp.iterImages(split = 'test', max_images = max_images):
    n+=1
    print 'image %d/%d:' % (n, max_images)
    references = [' '.join(x['tokens']) for x in img['sentences']] # as list of lists of tokens
    kwparams = { 'beam_size' : params['beam_size'] }
    Ys = BatchGenerator.predict([{'image':img}], model, checkpoint_params, **kwparams)

    img_blob = {} # we will build this up
    img_blob['img_path'] = img['local_file_path']
    img_blob['imgid'] = img['imgid']

    if dump_folder:
      # copy source file to some folder. This makes it easier to distribute results
      # into a webpage, because all images that were predicted on are in a single folder
      source_file = img['local_file_path']
      target_file = os.path.join(dump_folder, os.path.basename(img['local_file_path']))
      os.system('cp %s %s' % (source_file, target_file))

    # encode the human-provided references
    img_blob['references'] = []
    for gtsent in references:
      print 'GT: ' + gtsent
      img_blob['references'].append({'text': gtsent})

    # now evaluate and encode the top prediction
    top_predictions = Ys[0] # take predictions for the first (and only) image we passed in
    top_prediction = top_predictions[0] # these are sorted with highest on top
    candidate = ' '.join([ixtoword[ix] for ix in top_prediction[1] if ix > 0]) # ix 0 is the END token, skip that
    print 'PRED: (%f) %s' % (top_prediction[0], candidate)

    # save for later eval
    all_references.append(references)
    all_candidates.append(candidate)

    img_blob['candidate'] = {'text': candidate, 'logprob': top_prediction[0]}    
    blob['imgblobs'].append(img_blob)

  # use perl script to eval BLEU score for fair comparison to other research work
  # first write intermediate files
  print 'writing intermediate files into eval/'
  open('eval/output', 'w').write('\n'.join(all_candidates))
  for q in xrange(5):
    open('eval/reference'+`q`, 'w').write('\n'.join([x[q] for x in all_references]))
  # invoke the perl script to get BLEU scores
  print 'invoking eval/multi-bleu.perl script...'
  owd = os.getcwd()
  os.chdir('eval')
  os.system('./multi-bleu.perl reference < output')
  os.chdir(owd)

  # now also evaluate test split perplexity
  gtppl = eval_split('test', dp, model, checkpoint_params, misc, eval_max_images = max_images)
  print 'perplexity of ground truth words based on dictionary of %d words: %f' % (len(ixtoword), gtppl)
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
  parser.add_argument('-d', '--dump_folder', type=str, default="", help='dump the relevant images to a separate folder with this name?')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
