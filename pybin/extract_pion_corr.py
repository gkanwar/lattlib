#!/usr/bin/env python

## Extracts pion correlator info from propagator inversion output xml.

import pickle
import re
import sys

def usage(name):
    print "Usage: %s corr_out.np cfg1 prop1.out.xml [cfg2 prop2.out.xml ...]" % name

if __name__ == "__main__":
    # Check arg length
    if len(sys.argv) < 4:
        usage(sys.argv[0])
        sys.exit(1)
    if len(sys.argv) % 2 != 0:
        usage(sys.argv[0])
        sys.exit(2)
    # Extract args
    out_filename = sys.argv[1]
    cfgs = map(int, sys.argv[2::2])
    prop_xml_filenames = sys.argv[3::2]

    # Output mapping t to dict mapping cfg to [[corr]]
    out_dict = {}

    prop_corr_patt = re.compile(r"\s*<sink_smeared_prop_corr>([^<]*)</sink_smeared_prop_corr>\s*")
    for cfg,prop_xml_filename in zip(cfgs, prop_xml_filenames):
        print "Reading", prop_xml_filename
        try:
            with open(prop_xml_filename, 'r') as f:
                for line in f:
                    match = prop_corr_patt.match(line)
                    if match is None: continue
                    corrs = map(float, match.group(1).strip().split(" "))
                    for t,corr in enumerate(corrs):
                        if t in out_dict:
                            out_dict[t][cfg] = [[corr]]
                        else:
                            out_dict[t] = {cfg: [[corr]]}
                    break
        except IOError:
            print "WARNING: Could not find file %s, skipping." % (prop_xml_filename)

    # Write the generated dict to given np output
    print "Writing output to", out_filename
    with open(out_filename, 'wb') as f:
        pickle.dump(out_dict, f)
