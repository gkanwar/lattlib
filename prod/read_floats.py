#!/usr/bin/env python

"""
TODO: Merge with read_doubles.py
Read a file and extract the first n raw (single) floats.
Useful for examining packed propagator and gauge
slice files.
"""

from __future__ import print_function
import struct
import sys

def usage():
    print("Usage: %s (big|little) n file" % (sys.argv[0],))

if len(sys.argv) < 4:
    usage()
    sys.exit(-1)
if sys.argv[1] not in ["big", "little"]:
    usage()
    print("%s not one of (big|little)" % (sys.argv[1],))
    sys.exit(-1)

endianChar = (">" if sys.argv[1] == "big" else "<")
ndoubles = int(sys.argv[2])
filename = sys.argv[3]
print("Reading %d %s-endian doubles from %s" % (ndoubles,endianChar,filename),
      file=sys.stderr)
fmt = '%s%df' % (endianChar,ndoubles)
numbytes = struct.calcsize(fmt)
with open(filename, 'rb') as f:
    flts = struct.unpack(fmt, f.read()[:numbytes])
for d in flts:
    print(d)
