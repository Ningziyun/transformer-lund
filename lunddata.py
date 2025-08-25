import ROOT
import os
import fastjet
import awkward as ak
import math
import ljpHelpers


def loopFile(m_filename, tree, outdir="inputFiles", outname="qcd_lund.root", nImages=30, minDr=0.0, maxDr=10.0, minKt=-1, maxKt=8, minZ=0.5, maxZ=6.5, nBinsKt=25, nBinsDr=25, nBinsZ=40):
   # Set branch addresses and branch pointers
   if (not tree):
     return;
     
   if not os.path.exists(outdir):
     os.makedirs(outdir)

   # The output file with the tree
   newfile = ROOT.TFile.Open(os.path.join(outdir, outname), "RECREATE");
   # Output TTree and variables
   lundTree = ROOT.TTree("lundTree", "Jet declustering kt and deltaR")
   kt_vec = ROOT.std.vector('float')()
   deltaR_vec = ROOT.std.vector('float')()
   lundTree.Branch("kt", kt_vec)
   lundTree.Branch("deltaR", deltaR_vec)

   # Index for how many jets have been analyzed
   njet = 0;
   # Index for the event number
   jentry = 0;

   # Loop through all events in the input file to read information about the jets and constituents,
   # and use this to fill the tree with this data
   for index, event in enumerate(tree):
     #if index > 1000: 
     #  break
     jentry += 1

     # Read the kinematic information about the jet constituents from the tree
     constit_pt = event.constit_pt
     constit_eta = event.constit_eta
     constit_phi = event.constit_phi

     jetR10 = 1.0;
     jetDef10 = fastjet.JetDefinition(fastjet.antikt_algorithm, jetR10, fastjet.E_scheme);

     for cjet in range(len(constit_pt)):
       njet += 1;
       constituents = [];

       # Convert the constituent information into a format usable for fastjet (PseudoJet objects)
       for j in range(len((constit_pt))):
         constitTLV = ROOT.TLorentzVector(0, 0, 0, 0);
         constitTLV.SetPtEtaPhiM((constit_pt)[j], (constit_eta)[j], (constit_phi)[j], 0);
         constitPJ = fastjet.PseudoJet(constitTLV.Px(), constitTLV.Py(), constitTLV.Pz(), constitTLV.E());
         constituents.append(constitPJ);
         
     # Run the jet clustering on the jet constituents using the anti-kt algorithm
     clustSeq4 = fastjet.ClusterSequence(constituents, jetDef10);
     inclusiveJets10 = fastjet.sorted_by_pt(clustSeq4.inclusive_jets(25.));

     # Skip if inclusiveJets10 is empty
     if not inclusiveJets10:
       continue

     # Recluster the jets using the Cambridge-Aachen algorithm
     allConstits = list(inclusiveJets10[0].constituents())
     cs_ca = fastjet.ClusterSequence(allConstits, fastjet.JetDefinition1Param(fastjet.cambridge_algorithm, 10.0));
     myJet_ca = fastjet.sorted_by_pt(cs_ca.inclusive_jets(1.0));

     # Get Lund plane declusterings
     lundPlane = ljpHelpers.jet_declusterings(inclusiveJets10[0]);

     for k in range(len(lundPlane)):
       # Fill the tree with declustered information
       if (lundPlane[k].delta_R > 0 and lundPlane[k].z > 0):       
         deltaR_vec.push_back(lundPlane[k].delta_R)
         kt_vec.push_back(lundPlane[k].kt)

     if len(kt_vec) > 0:
        lundTree.Fill()
        kt_vec.clear()
        deltaR_vec.clear()
      # Normalize by the number of jets analyzed

   # Write the tree to the output file
   newfile.cd()
   lundTree.Write()
   newfile.Close();


import argparse
parser = argparse.ArgumentParser(description='Process benchmarks.')
parser.add_argument("--filename", help="", default="fileList.txt")
parser.add_argument("--treename", help="", default="tree")

opt = parser.parse_args()


if not os.path.exists("rootFiles"):
    os.makedirs("rootFiles")

with open(opt.filename) as infile:
  for line in infile:

    # Skip commented lines
    if(line[0] == '#'):
      continue;
    line = line.rstrip('\n')

    tree = ROOT.TTree();

    try:
      file = ROOT.TFile(line);
      tree = file.Get(opt.treename);
    except:
      file = None
      print("Did not find either file or tree, continuing to the next")
      continue

    if(not tree):
      continue;

    # Always write to inputFiles/qcd_lund.root
    loopFile("ignored.root", tree);
    file.Close();
