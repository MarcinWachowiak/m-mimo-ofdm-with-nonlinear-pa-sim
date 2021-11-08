#optional numba decorator
try:
    from numba import jit
except:
    def jit(pyfunc=None, **kwargs):
        def wrap(func):
            return func
        if pyfunc is not None:
            return wrap(pyfunc)
        else:
            return wrap