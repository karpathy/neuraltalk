from random import uniform
import numpy as np

def randi(N):
  """ get random integer in range [0, N) """
  return int(uniform(0, N))

def merge_init_structs(s0, s1):
  """ merge struct s1 into s0 """
  for k in s1['model']:
    assert (not k in s0['model']), 'Error: looks like parameter %s is trying to be initialized twice!' % (k, )
    s0['model'][k] = s1['model'][k] # copy over the pointer
  s0['update'].extend(s1['update'])
  s0['regularize'].extend(s1['regularize'])

def initw(n,d): # initialize matrix of this size
  magic_number = 0.1
  return (np.random.rand(n,d) * 2 - 1) * magic_number # U[-0.1, 0.1]

def accumNpDicts(d0, d1):
  """ forall k in d0, d0 += d1 . d's are dictionaries of key -> numpy array """
  for k in d1:
    if k in d0:
      d0[k] += d1[k]
    else:
      d0[k] = d1[k]