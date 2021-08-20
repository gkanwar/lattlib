#!/usr/bin/env python

## Extract lattice complex from a chroma output file and write all
## resulting complexes in packed format into given .dat file.

import re
import struct
import sys

if len(sys.argv) < 3:
    print "Usage: %s <chroma.out.xml> <packed.dat>" % sys.argv[0]
    sys.exit(1)

in_xml = open(sys.argv[1], 'r')
out_dat = open(sys.argv[2], 'wb')

re_patt = re.compile(r"\s*<re>([^<]+)</re>\s*")
im_patt = re.compile(r"\s*<im>([^<]+)</im>\s*")
ol_open_patt = re.compile(r"\s*<OLattice>\s*")
ol_close_patt = re.compile(r"\s*</OLattice>\s*")

inside_ol = False
lattice_complex_count = 0
re = None
for line in in_xml:
    # Look for <re>
    m = re_patt.match(line)
    if m is not None:
        if not inside_ol: continue
        re = float(m.group(1))
        continue
    # Look for <im>
    m = im_patt.match(line)
    if m is not None:
        if not inside_ol: continue
        assert re is not None
        im = float(m.group(1))
        out_dat.write(struct.pack('dd', re, im))
        re = None
        continue
    # Look for <OLattice>
    m = ol_open_patt.match(line)
    if m is not None:
        assert not inside_ol
        inside_ol = True
        print "Found open OLattice"
        continue
    # Look for </OLattice>
    m = ol_close_patt.match(line)
    if m is not None:
        assert inside_ol
        inside_ol = False
        lattice_complex_count += 1
        print "Found close OLattice"
        continue

print "Wrote %d lattice complexes into %s." % (lattice_complex_count, sys.argv[2])
