### Some utilities to interact with USQCD data.

import struct
import numpy as np

# Extract a single record in binary format from a LIME file.
# See https://usqcd-software.github.io/c-lime/lime_1p2.pdf for details.
# Can be used to pull raw propagators or gauge fields out in a handy numpy
# format for simple manipulations.
LIME_HEADER_SIZE = 144
LIME_MAGIC = 0x456789ab
LIME_MB_BIT = 15
LIME_ME_BIT = 14
def extract_lime_records(filename, msg_record_inds, *, verbose=False):
    cur_msg = 1
    cur_record = 1
    outputs = [None]*len(msg_record_inds)
    seen = 0

    with open(filename, 'rb') as f:
        if verbose: print(f'Reading LIME {filename}')
        while seen < len(outputs):
            # read and parse header
            header = f.read(LIME_HEADER_SIZE)
            if len(header) != LIME_HEADER_SIZE:
                raise RuntimeError('Failed to find all LIME records before EOF')
            magic, version, flags, data_len, lime_type = struct.unpack('>LHHQ128s', header)
            if magic != LIME_MAGIC:
                raise RuntimeError(f'File is not in LIME format (bad magic {magic})')
            if version != 1:
                raise RuntimeError(f'LIME version {version} not supported')
            message_begin = (flags >> LIME_MB_BIT) & 1
            message_end = (flags >> LIME_ME_BIT) & 1
            pad = (-data_len) % 8
            if verbose:
                print(f'... message {cur_msg} record {cur_record}')
                print(f'... raw flags {flags:016b}')
                print(f'... MB = {message_begin} ME = {message_end}')
                print(f'... len {data_len} pad {pad}')

            # either read data bytes or skip them
            if (cur_msg, cur_record) in msg_record_inds:
                data = f.read(data_len)
                f.seek(pad, 1)
                if len(data) != data_len:
                    raise RuntimeError('Data read failed')
                outputs[msg_record_inds.index((cur_msg, cur_record))] = data
                seen += 1
            else:
                f.seek(data_len + pad, 1)

            # update message / record numbers
            if message_begin:
                assert cur_record == 1, 'inconsistent MB bit'
            if message_end:
                cur_msg += 1
                cur_record = 1
            else:
                cur_record += 1

    return outputs


# Extract prop from Chroma format
def extract_lime_propagator(filename, latt_shape, *, Nc=3, Ns=4, prec='double'):
    Nd = len(latt_shape)
    # TODO: replace explicit shape with inspecting format_xml
    format_xml, data = extract_lime_records(filename, [(2,1), (2,3)])
    prop_shape = tuple(reversed(latt_shape)) + (Ns, Ns, Nc, Nc)
    assert prec in ['double', 'single']
    in_dtype = '>c16' if prec == 'double' else '>c8'
    out_dtype = np.complex128 if prec == 'double' else np.complex64
    arr = np.frombuffer(data, dtype=in_dtype).reshape(prop_shape).astype(out_dtype)
    inds = list(reversed(range(Nd))) + [Nd, Nd+1, Nd+2, Nd+3]
    return np.transpose(arr, inds)

# Extract cfg from default Chroma format (ILDG)
def extract_lime_cfg(filename, latt_shape, *, Nc=3, prec='double'):
    Nd = len(latt_shape)
    # TODO: replace explicit shape with inspecting format_xml
    format_xml, data = extract_lime_records(filename, [(2,3), (2,4)])
    cfg_shape = (Nd,) + tuple(reversed(latt_shape)) + (Nc,Nc)
    assert prec in ['double', 'single']
    in_dtype = '>c16' if prec == 'double' else '>c8'
    out_dtype = np.complex128 if prec == 'double' else np.complex64
    arr = np.frombuffer(data, dtype=in_dtype).reshape(cfg_shape).astype(out_dtype)
    inds = [0] + list(reversed(range(1,Nd+1))) + [Nd+1, Nd+2]
    return np.transpose(arr, inds)
