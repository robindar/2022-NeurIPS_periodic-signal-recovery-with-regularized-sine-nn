import sys
import numpy as numpy

numpy.set_printoptions(threshold=sys.maxsize)
serialize_nparray = lambda a: str(a.shape) + "!" + numpy.array2string(a.reshape(-1), separator=",", max_line_width=numpy.inf)
serialize_tuple = lambda tup: list(serialize_nparray(numpy.array(v)) for v in tup)
serialize_dict = lambda dictionnary: { k: serialize_nparray(numpy.array(v)) for k,v in dictionnary.items() }

def deserialize_array(d):
    shape, serialized = d.split('!')
    shape = shape[1:-1].split(',')
    shape = tuple(int(x) for x in shape if x != '')
    return numpy.fromstring(serialized[1:-1], sep=',').reshape(shape)

deserialize_tuple = lambda t: tuple(deserialize_array(x) for x in t)

def serialize_nested(u):
    if isinstance(u, tuple):
        return list(serialize_nested(v) for v in u)
    elif isinstance(u, list):
        return list(serialize_nested(v) for v in u)
    elif isinstance(u, dict):
        return { k: serialize_nested(v) for k, v in u.items() }
    else:
        return serialize_nparray(u)

def deserialize_nested(u):
    if isinstance(u, list):
        return list(deserialize_nested(v) for v in u)
    elif isinstance(u, dict):
        return { k: deserialize_nested(v) for k, v in u.items() }
    else:
        return deserialize_array(u)
