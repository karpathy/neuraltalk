import time
import numpy as np
from imagernn.utils import randi

class Solver:
  """
  solver worries about:
  - different optimization methods, updates, weight decays
  - it can also perform gradient check
  """
  def __init__(self):
    self.step_cache_ = {} # might need this
    self.step_cache2_ = {} # might need this

  def step(self, batch, model, cost_function, **kwargs):
    """ 
    perform a single batch update. Takes as input:
    - batch of data (X)
    - model (W)
    - cost function which takes batch, model
    """
    
    learning_rate = kwargs.get('learning_rate', 0.0)
    update = kwargs.get('update', model.keys())
    grad_clip = kwargs.get('grad_clip', -1)
    solver = kwargs.get('solver', 'vanilla')
    momentum = kwargs.get('momentum', 0)
    smooth_eps = kwargs.get('smooth_eps', 1e-8)
    decay_rate = kwargs.get('decay_rate', 0.999)

    if not (solver == 'vanilla' and momentum == 0):
      # lazily make sure we initialize step cache if needed
      for u in update:
        if not u in self.step_cache_: 
          self.step_cache_[u] = np.zeros(model[u].shape)
          if solver == 'adadelta':
            self.step_cache2_[u] = np.zeros(model[u].shape) # adadelta needs one more cache

    # compute cost and gradient
    cg = cost_function(batch, model)
    cost = cg['cost']
    grads = cg['grad']
    stats = cg['stats']

    # clip gradients if needed, simplest possible version
    # todo later: maybe implement the gradient direction conserving version
    if grad_clip > 0:
      for p in update:
        if p in grads:
          grads[p] = np.minimum(grads[p], grad_clip)
          grads[p] = np.maximum(grads[p], -grad_clip)

    # perform parameter update
    for p in update:
      if p in grads:

        if solver == 'vanilla': # vanilla sgd, optional with momentum
          if momentum > 0:
            dx = momentum * self.step_cache_[p] - learning_rate * grads[p]
            self.step_cache_[p] = dx
          else:
            dx = - learning_rate * grads[p]

        elif solver == 'rmsprop':
          self.step_cache_[p] = self.step_cache_[p] * decay_rate + (1.0 - decay_rate) * grads[p] ** 2
          dx = -(learning_rate * grads[p]) / np.sqrt(self.step_cache_[p] + smooth_eps)
        
        elif solver == 'adagrad':
          self.step_cache_[p] += grads[p] ** 2
          dx = -(learning_rate * grads[p]) / np.sqrt(self.step_cache_[p] + smooth_eps)

        elif solver == 'adadelta':
          self.step_cache_[p] = self.step_cache_[p] * decay_rate + (1.0 - decay_rate) * grads[p] ** 2
          dx = - np.sqrt( (self.step_cache2_[p] + smooth_eps) / (self.step_cache_[p] + smooth_eps) ) * grads[p]
          self.step_cache2_[p] = self.step_cache2_[p] * decay_rate + (1.0 - decay_rate) * (dx ** 2)

        else:
          raise Exception("solver %s not supported" % (solver, ))

        # perform the parameter update
        model[p] += dx

    # create output dict and return
    out = {}
    out['cost'] = cost
    out['stats'] = stats
    return out

  def gradCheck(self, batch, model, cost_function, **kwargs):
    """ 
    perform gradient check.
    since gradcheck can be tricky (especially with relus involved)
    this function prints to console for visual inspection
    """

    num_checks = kwargs.get('num_checks', 10)
    delta = kwargs.get('delta', 1e-5)
    rel_error_thr_warning = kwargs.get('rel_error_thr_warning', 1e-2)
    rel_error_thr_error = kwargs.get('rel_error_thr_error', 1)

    cg = cost_function(batch, model)

    print 'running gradient check...'
    for p in model.keys():
      print 'checking gradient on parameter %s of shape %s...' % (p, `model[p].shape`)
      mat = model[p]

      s0 = cg['grad'][p].shape
      s1 = mat.shape
      assert s0 == s1, 'Error dims dont match: %s and %s.' % (`s0`, `s1`)

      for i in xrange(num_checks):
        ri = randi(mat.size)

        # evluate cost at [x + delta] and [x - delta]
        old_val = mat.flat[ri]
        mat.flat[ri] = old_val + delta
        cg0 = cost_function(batch, model)
        mat.flat[ri] = old_val - delta
        cg1 = cost_function(batch, model)
        mat.flat[ri] = old_val # reset old value for this parameter

        # fetch both numerical and analytic gradient
        grad_analytic = cg['grad'][p].flat[ri]
        grad_numerical = (cg0['cost']['total_cost'] - cg1['cost']['total_cost']) / ( 2 * delta )

        # compare them
        if grad_numerical == 0 and grad_analytic == 0:
          rel_error = 0 # both are zero, OK.
          status = 'OK'
        elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
          rel_error = 0 # not enough precision to check this
          status = 'VAL SMALL WARNING'
        else:
          rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
          status = 'OK'
          if rel_error > rel_error_thr_warning: status = 'WARNING'
          if rel_error > rel_error_thr_error: status = '!!!!! NOTOK'

        # print stats
        print '%s checking param %s index %8d (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' \
              % (status, p, ri, old_val, grad_analytic, grad_numerical, rel_error)








