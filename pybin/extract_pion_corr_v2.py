#!/usr/bin/env python

## Extracts pion correlator from propagator inversion output xml and returns
## a packed complexes ".dat" file, with Nmeas x Nt elements.

from read_helper import write_lattice_complexes
from xml.dom.minidom import *
import numpy as np
import sys

def usage(name):
    print "Usage: %s out.dat prop1.out.xml [prop2.out.xml ...]" % name

def main(out_filename, prop_xml_filenames):
    all_out = []
    for fname in prop_xml_filenames:
        print "Reading %s..." % fname
        dom = parse(fname)
        corr_node = dom.getElementsByTagName("sink_smeared_prop_corr")[0]
        corrs = map(float, corr_node.firstChild.data.strip().split(' '))
        all_out.append(corrs)
    # Cast to complex, for standardization
    all_out = np.array(all_out, dtype=np.complex128)
    write_lattice_complexes(all_out, out_filename)

if __name__ == "__main__":
    # Check arg length
    if len(sys.argv) < 3:
        usage(sys.argv[0])
        sys.exit(1)
    # Extract args
    out_filename = sys.argv[1]
    prop_xml_filenames = sys.argv[2:]
    main(out_filename, prop_xml_filenames)
