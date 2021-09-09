# import paddle
import torch
import shared

# Reverse mode AD
# 
# Key concepts:
#   
#   - adjoint function: for any function *fun* to be differentiated, the adjoint function
#        refers the generated function that carries out the actual differentiation work.
#   - continuation: a function abstraction representing the residual work for some
#        initial computation. A continuation doesn't return but it simply ends by calling
#        into its own continuation.
#   - vjp: the basic building block of reverse mode AD. It takes in a function *fun* and
#        a list of primals to evaluate *fun* at then returns a tuple (fun_out, vjp_fun).
#        fun_out is the result of fun(*primals) while vjp_fun(), given a vector v, returns
#        the vector-jacobian product.
# 
# To demonstrate the idea of reverse mode AD, consider that following code example.
#
# def fancy_layer(x, y, W, B):
#   u = matmul(x, W)
#   s = tanh(u + B)
#   return dot(s, y)
#
# The reverse mode AD is defined as constructing the adjoint function vjp for input fun.
# Since we are not resorting to source transformation, we likely end up with something
# like a function wrapper.
# 
# The logic of reverse mode AD subsumes a forward pass, ie. tracing of the function,
# though during the forward pass it's not necessary to perform the actual computation. 
# It suffices to get enough information for us to generate the adjoint vjp.
# 
# We treat the generation of the adjoint function as a procedure of building up a chain
# of CPS nodes.
#
#
#       primal                 adjoint
#  ----------------      -------------------
#                        promises = {}
#                        promises.add('x', Promise(x)); promises.add('y', Promise(y)); k0 = end_k(aggregates)
#   u = matmul(x, W)     promises.add('u', Promise(u)); k1 = vjp_k(matmul, x, promises('u'), k0)
#   s = tanh(u + B)      promises.add('s', Promise(s)); k2 = vjp_k(tanh, u, promises('s'), k1)
#   AD.__y = dot(s, y)   promises.add('AD.__y', Promises(AD.__y)); k3 = vjp_k(dot, s, y, promises('AD.__y'), k2)
#                        def vjp_fn(v):
#                          promises.set('AD.__y', v)
#                          return k3()
#                        return AD.__y, vjp_fn
# 

OpStrength = {'const', 'linear', 'poly', 'nonlinear'}

stren_tab = {'const': [],
             'linear': [
                'elementwise_add',
                'elementwise_mul',
                'elementwise_sub',
                'matmul_V2'
             ],
             'poly': [
                'elementwise_pow' 
             ],
             'nonlinear': [
                'exp',
                'tanh'
             ]}

# stren_lookup = {}
# op_vjps = {}

# def build_strenlookup():
#   stren_lookup.update((op, stren) for stren, op in stren_tab.items())

class OpVJPs:
  vjp_mappings = {}

  @classmethod
  def defvjp(cls, fn, *makers):
    OpVJPs.vjp_mappings[fn] = makers

  @classmethod
  def get_vjpmakers(cls, fn):
    # print(OpVJPs.vjp_mappings)
    return OpVJPs.vjp_mappings[fn]

class tracing_context:
  def __init__(self):
    self.grads = {}
    self.k = None

  def get_closure(self):
    return self.k

  def set_closure(self, k):
    self.k = k

  def make_vjp_k(self, vjp, k, x_pos, y, *xs):
    def vjp_k():
      vjp_fn = vjp(y, *xs)
      print(f'[Reverse op begin] {vjp_fn}')
      dx = vjp_fn(self.grads[y])
      self.grads[xs[x_pos]] = dx
      print(f'[Reverse op end  ] {vjp_fn}')
      print(f'[Next reverse ...] {k}')
      return k()
    return vjp_k

def vjp(f, *xs, y_argnum=0):
  tc_stack = shared.tc_stack

  tc = tracing_context()
  tc_stack.append(tc)

  print(f'[VJP start] {f}')

  def return_grads():
    return [tc.grads[x] for x in xs]

  tc.k = return_grads

  for x in xs:
    tc.grads[x] = None

  # The tc stack grows as higher order of grads are building up the computation graph.
  
  res = f(*xs)
  
  if isinstance(res, list):
    res = res[y_argnum]

  def vjp_fn(dy):
    tc.grads[res] = dy
    k = tc.get_closure()
    print(f'[VJP_FN for {f}] == closure in {tc}: {k}')
    return k()

  shared.tc_stack.pop()
  print(f'[VJP end  ] {f}')

  return res, vjp_fn

def grad(f):
  def grad_f(*xs):
    res, vjp_fn = vjp(f, *xs)
    while isinstance(res, list):
      res = res[0]
    return vjp_fn(torch.ones_like(res))
  return grad_f

if __name__ == '__main__':

  # x = paddle.rand([2])
  # v = paddle.rand([2])
  x = torch.rand(2)
  v = torch.rand(2)

  from wrapper import exp, tanh, mul
  # print(exp)
  def foo(x):
    return exp(x)

  res, vjp_fun = vjp(foo, x)
  print(vjp_fun(v))
  f_x = grad(foo)
  print(f'f_x(x) = {f_x(x)}')

  f_xx = grad(grad(foo))
  print(f'f_xx = {f_xx}')
  print(f'f_xx(x) = {f_xx(x)}')

  f_xxx = grad(grad(grad(foo)))
  print(f'f_xxx(x) = {f_xxx(x)}')