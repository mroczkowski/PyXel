import inspect

def const(constant=1e-2):
    return constant

print(inspect.getcallargs(const).keys())
args, _junk, _junk, def_vals = inspect.getargspec(const)
print(args)
if hasattr(const, 'constant'):
    print('say hello')
