#!/usr/bin/env python

## Script to extract the prop_corr info from an HMC stream output, assuming you
## have run PROPAGATOR measurements inline. Uses sequential order to label the
## cfgs.

from xml.dom.minidom import *
import pickle
import sys

def usage(name):
    print "Usage: %s <hmc.out.xml> <out.np>" % name

if __name__ == "__main__":
    # Check arg length
    if len(sys.argv) < 3:
        usage(sys.argv[0])
        sys.exit(1)
    # Extract args
    xml_log_filename = sys.argv[1]
    out_filename = sys.argv[2]

    d = parse(xml_log_filename)
    out_dict = {}
    cfg = 1
    for n in d.getElementsByTagName("prop_corr"):
        p = n.childNodes[0].data
        corrs = map(float, p.strip().split(' '))
        for t,corr in enumerate(corrs):
            if t in out_dict:
                out_dict[t][cfg] = [[corr]]
            else:
                out_dict[t] = {cfg: [[corr]]}
        cfg += 1

    # Write the generated dict to given np output
    print "Writing output to", out_filename
    with open(out_filename, 'wb') as f:
        pickle.dump(out_dict, f)
