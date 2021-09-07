import paddle

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
                myops.add,
                myops.multiply,
                myops.subtract,
                myops.matmul
             ],
             'poly': [
                myops.power  
             ],
             'nonlinear': [
                myops.exp,
                myops.tanh
             ] }

stren_lookup = {}
op_vjps = {}

def build_strenlookup():
  stren_lookup.update((op, stren) for stren, op in stren_tab.items())

def defvjp(fn, *makers):
  op_vjps[fn] = makers

defvjp(myops.tanh, lambda x, y: lambda v: v - v * y**2)
defvjp(myops.exp, lambda x, y: lambda v: v * y)

tc_stack = []

class tracing_context:
  def __init__(self):
    self.promises = {}
    self.kstack = []  

  def make_vjp_k(self, vjp, x_pos, y, *xs):
    def vjp_k(k):
      vjp_fn = vjp(*xs)
      dx = vjp_fn(self.promises[y])
      self.promises[xs[x_pos]] = dx
      return k()
    return vjp_k
