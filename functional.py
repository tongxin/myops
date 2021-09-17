import paddle
from typing_extensions import ParamSpecArgs
import contextlib
import functional
import typing

class Map:
  def __init__(self) -> None:
    pass
  def __call__(self, *args: Any, **kwds: Any) -> Any:
    pass

class LinearMap(Map):
  pass

# overloading on y. If y is callable, then return a vjp which is
# the differential of y at x. If y is variable, then return the
# jacobian.
def diff_rev(y, x=None):
  pass

# returns the differential of f 
def differential(f):
  def d(x):
    y = f(x)
    return diff_rev(y, x)
  return d


@contextlib.contextmanager
def gradient_scope(vars, open_scope=False):
  xs = []
  try:
    if open_scope:
      # If v is treated as constant in the outer scope, its gradient is guaranteed
      # not to be taken beyond this scope. Within this scope, however, v's gradient
      # may be computed. We only need to detach v in this case.
      # Otherwise, v's gradient is valid, and is subject to update beyond this scope.
      # In this case we must not confuse the gradient in the outer scope with the
      # inner one's. Moreover, we need to make sure that the result from the inner
      # scope can flow back to the outer scope. This can be satisfied by extending
      # the original variable with a duplication operation v1 = v so that v still
      # maintains the complete lineage.
      for v in vars:
        if v.stop_gradient:
          v = v.detach()
          v.stop_gradient = False
        else:
          v = paddle.assign(v)
        xs.append(v)
    else:
      # Results won't flow back to outer scope so just detach the variables
      for v in vars:
        xs.append(v.detach())
    yield lambda ys, xs, v: paddle.grad(ys, xs, v, create_graph=open_scope), xs
  finally:
    pass


def vjp(func, inputs, v, grad_results=False):

  with gradient_scope(inputs, open_scope=grad_results) as (grad_fn, xs):
    ys = func(*xs)
    vjps = grad_fn(ys, xs, v)
  
  return vjps
