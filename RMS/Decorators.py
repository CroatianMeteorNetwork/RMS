""" Module with commonly used decorators. """


import functools
import inspect


class memoizeAll(object):
    """ Decorator. Caches a function's return value each time it is called. If called later with the same 
        arguments, the cached value is returned (not reevaluated). 

        This version caches all inputs.

    Source: https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}


    def __call__(self, *args):

        # Check if arguments already cached
        if args in self.cache:
            return self.cache[args]

        # If not, compute the function value and store in cache
        else:
            value = self.func(*args)
            self.cache[args] = value

            return value


    def __repr__(self):
        """ Return the function's docstring. """

        return self.func.__doc__


    def __get__(self, obj, objtype):
        """ Support instance methods. """

        return functools.partial(self.__call__, obj)




class memoizeSingle(object):
    """ Decorator. Caches a function's return value each time it is called. If called later with the same 
        arguments, the cached value is returned (not reevaluated). 

        This version caches only one input and resets every time a different input is given.

    Source: https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}


    def __call__(self, *args, **kwargs):

        key = self.key(args, kwargs)

        # Check if arguments already cached
        if key in self.cache:
            return self.cache[key]

        # If not, compute the function value and store in cache
        else:

            # Reset the cache
            self.cache = {}

            # Compute the function value
            value = self.func(*args, **kwargs)

            # Store the compute value in cache
            self.cache[key] = value

            return value


    def __repr__(self):
        """ Return the function's docstring. """

        return self.func.__doc__


    def __get__(self, obj, objtype):
        """ Support instance methods. """

        return functools.partial(self.__call__, obj)

    def normalize_args(self, args, kwargs):
        spec = inspect.getargs(self.func.__code__).args
        return dict(list(kwargs.items()) + list(zip(spec, args)))

    def key(self, args, kwargs):
        a = self.normalize_args(args, kwargs)
        return tuple(sorted(a.items()))