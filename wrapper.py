import paddle

op_names = ['add', 'matmul', 'exp', 'tanh']

def wrap_paddle_op(op, op_name):
  vjp_makers = op_vjps[op_name]

  def wrapped_op(*args):
    if not tc_stack:
      return op(*args)

    tc = tc_stack[-1]
    # positional params to track the grads on
    x_pos_s = [pos for pos, x in enumerate(args) if x in tc.promises]

    for x_pos in x_pos_s:
      vjp = vjp_makers[x_pos]
      k = tc.make_vjp_k(vjp, x_pos, y, *args) ## how to plugin y?
      tc.kstack.append(k)
      
    res = op(*args)
    
    return res

  return wrapped_op