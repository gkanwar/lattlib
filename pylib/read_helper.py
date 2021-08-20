from functools import reduce
import numpy as np
import operator
import os
import struct

def read_doubles(f, ndoubles, endian='<'):
    fmt = '%s%dd' % (endian,ndoubles)
    nbytes = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read()[:nbytes])

def write_doubles(dbls, f, endian='<'):
    fmt = '%s%dd' % (endian,dbls.size)
    f.write(struct.pack(fmt, *dbls.flatten()))

def read_lattice_doubles(filename, dims):
    size = reduce(operator.mul, dims, 1)
    with open(filename, 'rb') as f:
        dbls = read_doubles(f, size)
    arr = np.array(dbls)
    return arr.reshape(dims)

def read_lattice_complexes(filename, dims):
    size = reduce(operator.mul, dims, 1)
    with open(filename, 'rb') as f:
        dbls = read_doubles(f, size*2) # 2 doubles per complex
    arr_re = np.array(dbls[0::2])
    arr_im = np.array(dbls[1::2])
    arr = arr_re + np.complex(0,1)*arr_im
    return arr.reshape(dims)

def write_lattice_doubles(dbls, filename, mode='wb'):
    with open(filename, mode) as f:
        write_doubles(dbls, f)

def write_lattice_complexes(cs, filename, mode='wb'):
    cs_flat = cs.flatten()
    dbls = np.zeros(cs_flat.size*2, dtype=np.float64)
    dbls[0::2] = np.real(cs_flat)
    dbls[1::2] = np.imag(cs_flat)
    write_lattice_doubles(dbls, filename, mode)

def read_all_doubles(f, endian='<'):
    elt_fmt = '%sd' % endian
    elt_size = struct.calcsize(elt_fmt)
    bs = f.read()
    ndoubles = len(bs) // elt_size
    fmt = '%s%dd' % (endian,ndoubles)
    nbytes = struct.calcsize(fmt)
    assert nbytes == len(bs)
    return struct.unpack(fmt, bs)

def read_all_complexes(filename):
    with open(filename, 'rb') as f:
        dbls = read_all_doubles(f)
    arr_re = np.array(dbls[0::2])
    arr_im = np.array(dbls[1::2])
    arr = arr_re + np.complex(0,1)*arr_im
    return arr
