from functools import wraps
# import paddle
# from paddle.fluid.core import ops
import torch
import ad
import shared

# op_names = [
#             'elementwise_add',
#             'ops.elementwise_sub',
#             'elementwise_mul',
#             'matmul_v2',
#             'elementwise_pow']

op_names = [
            'exp',
            'tanh',
            'mul'
            ]

def exp(x):
  return torch.exp(x)
  # return paddle.exp(x)
  
def tanh(x):
  return torch.tanh(x)
  # return paddle.tanh(x)

def mul(x, y):
  return torch.mul(x, y)
  # return paddle.multiply(x, y)

# Wrap with tracing context
def check_tracing(op, op_name):
  vjp_makers = None

  @wraps(op)
  def wrapped(*args, **kwargs):
    nonlocal vjp_makers
    if not vjp_makers:
      vjp_makers = ad.OpVJPs.get_vjpmakers(op_name)

    tc_stack = shared.tc_stack

    print(f'[Forward op] {op}')
    res = op(*args, **kwargs)

    if not tc_stack:
      return res

    tc = tc_stack[-1]
    # positional params to track the grads on
    argnums = [argnum for argnum, x in enumerate(args) if x in tc.grads]

    print(f'[Building reverse] {op}')  
    for i in argnums:
      vjp = vjp_makers[i]
      tc.set_closure(tc.make_vjp_k(vjp, tc.k, i, res, *args))
      print(f'  |___ built in closure {tc.get_closure()} in tc: {tc} ')

    tc.grads[res] = None

    return res

  return wrapped


def wrap_ops():
  g = globals()
  for op_name in op_names:
    assert op_name in g
    op = check_tracing(g[op_name], op_name)
    g[op_name] = op

wrap_ops()

ad.OpVJPs.defvjp('tanh', lambda y, x: lambda v: v - v * y**2)
# OpVJPs.defvjp(exp, lambda y, x: lambda v: v * y)
ad.OpVJPs.defvjp('exp', lambda y, x: lambda v: mul(v, exp(x)))
ad.OpVJPs.defvjp('mul', lambda y, x1, x2: lambda v: mul(v, x2),
                        lambda y, x1, x2: lambda v: mul(v, x1))