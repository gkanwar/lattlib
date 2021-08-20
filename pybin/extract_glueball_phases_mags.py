#!/usr/bin/env python

## Extract the phases and mags from a glueball output file.

import numpy as np
import struct
import sys

if len(sys.argv) < 2:
    print "Usage: %s <dir>" % (sys.argv[0])
    sys.exit(1)

directory = sys.argv[1]

def split_xml_toks(s):
    return s.replace('>', '<').split('<')
        
f_phase = None # File to write phases into
f_mag = None # File to write mags into
i = None # left side mag index
j = None # right side mag index
disp = None # displacement structure
re = None # last real value - "look-behind"
for ind,line in enumerate(sys.stdin):
    if ind % 10000 == 0: print ind
    sline = line.strip()
    if sline == "<SpatialOp>":
        assert f_phase is None
        assert f_mag is None
        assert i is None
        assert j is None
        assert disp is None
    elif sline.startswith("<left>"):
        tokens = split_xml_toks(sline)
        assert tokens[1] == "left"
        assert tokens[3] == "/left"
        assert i is None
        i = int(tokens[2].strip())
    elif sline.startswith("<right>"):
        tokens = split_xml_toks(sline)
        assert tokens[1] == "right"
        assert tokens[3] == "/right"
        assert j is None
        j = int(tokens[2].strip())
    elif sline.startswith("<disp>"):
        tokens = split_xml_toks(sline)
        assert tokens[1] == "disp"
        assert tokens[3] == "/disp"
        assert disp is None
        disp = map(int, tokens[2].strip().split())
    elif sline == "<corr>":
        if disp is None: continue
        assert i is not None
        assert j is not None
        assert disp is not None
        assert f_phase is None
        assert f_mag is None
        disp_name = "disp[" + "".join(map(lambda d: ("n" if d < 0 else "") + str(abs(d)), disp)) + "]"
        f_phase = open("%s/%s_%d_%d_phase.dat" % (directory, disp_name, i, j), 'wb')
        f_mag = open("%s/%s_%d_%d_mag.dat" % (directory, disp_name, i, j), 'wb')
    elif sline.startswith("<re>"):
        if f_phase is None: continue
        assert re is None
        tokens = split_xml_toks(sline)
        assert tokens[1] == "re"
        assert tokens[3] == "/re"
        re = float(tokens[2].strip())
    elif sline.startswith("<im>"):
        if f_phase is None: continue
        assert re is not None
        tokens = split_xml_toks(sline)
        assert tokens[1] == "im"
        assert tokens[3] == "/im"
        im = float(tokens[2].strip())
        val = np.complex(re, im)
        re = None
        phase = np.angle(val)
        mag = np.abs(val)
        f_phase.write(struct.pack('d', phase))
        f_mag.write(struct.pack('d', mag))
    elif sline == "</SpatialOp>":
        f_phase = None
        f_mag = None
        i = None
        j = None
        disp = None
