# optional numba decorator
try:
    from numba import jit
except ImportError as error:
    print("Numba JIT module not found, resorting to standard interpreter!")


    def jit(pyfunc=None, **kwargs):
        """
        Wrapper/Decorator to use Numba JIT compiler to speed up the computations.

        """
        def wrap(func):
            return func

        if pyfunc is not None:
            return wrap(pyfunc)
        else:
            return wrap
