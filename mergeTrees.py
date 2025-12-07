#!/usr/bin/env python

import ROOT
import sys

inputs = sys.argv[2:]
output = sys.argv[1]
print(f"Merging {inputs} to {output}")
#ROOT.TFile.Merge(output, ROOT.std.vector("string")(inputs))

# Equivalent to: hadd -f output.root input1.root input2.root ...
merger = ROOT.TFileMerger()
merger.OutputFile(output, "RECREATE")
for inp in inputs: merger.AddFile(inp)
if not merger.Merge():
    print("Merging failed!")
else:
    print("Successfully merged files into merged.root")
