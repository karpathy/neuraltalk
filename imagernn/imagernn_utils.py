from imagernn.generic_batch_generator import GenericBatchGenerator
import numpy as np

def decodeGenerator(params):
  """ 
  in the future we may want to have different classes
  and options for them. For now there is this one generator
  implemented and simply returned here.
  """ 
  return GenericBatchGenerator

def eval_split(split, dp, model, params, misc, **kwargs):
  """ evaluate performance on a given split """
  # allow kwargs to override what is inside params
  eval_batch_size = kwargs.get('eval_batch_size', params.get('eval_batch_size',100))
  eval_max_images = kwargs.get('eval_max_images', params.get('eval_max_images', -1))
  BatchGenerator = decodeGenerator(params)
  wordtoix = misc['wordtoix']

  print 'evaluating %s performance in batches of %d' % (split, eval_batch_size)
  logppl = 0
  logppln = 0
  nsent = 0
  for batch in dp.iterImageSentencePairBatch(split = split, max_batch_size = eval_batch_size, max_images = eval_max_images):
    Ys, gen_caches = BatchGenerator.forward(batch, model, params, misc, predict_mode = True)

    for i,pair in enumerate(batch):
      gtix = [ wordtoix[w] for w in pair['sentence']['tokens'] if w in wordtoix ]
      gtix.append(0) # we expect END token at the end
      Y = Ys[i]
      maxes = np.amax(Y, axis=1, keepdims=True)
      e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
      P = e / np.sum(e, axis=1, keepdims=True)
      logppl += - np.sum(np.log2(1e-20 + P[range(len(gtix)),gtix])) # also accumulate log2 perplexities
      logppln += len(gtix)
      nsent += 1

  ppl2 = 2 ** (logppl / logppln) 
  print 'evaluated %d sentences and got perplexity = %f' % (nsent, ppl2)
  return ppl2 # return the perplexity
