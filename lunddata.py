import ROOT
import os
import fastjet
import awkward as ak
import math
import ljpHelpers


# Added logMode and swapAxes options
def loopFile(m_filename, tree, outdir="inputFiles", outname="qcd_lund.root",
             nImages=30, minDr=0.0, maxDr=10.0, minKt=-1, maxKt=8,
             minZ=0.5, maxZ=6.5, nBinsKt=25, nBinsDr=25, nBinsZ=40,
             logMode=False, swapAxes=False,mode="kt"):

   # Set branch addresses and branch pointers
   if (not tree):
     return;
     
   if not os.path.exists(outdir):
     os.makedirs(outdir)
   """
   # The output file with the tree
   # --- safer open mode: create if not exist, else append ---
   outpath = os.path.join(outdir, outname)
   if os.path.exists(outpath):
       newfile = ROOT.TFile.Open(outpath, "UPDATE")   # append to existing file
   else:
       newfile = ROOT.TFile.Open(outpath, "RECREATE") # create new file

   # Output TTree and variables   
   lundTree = ROOT.TTree("lundTree", "Jet declustering kt and deltaR")
   deltaR_vec = ROOT.std.vector('float')()
   val_vec = ROOT.std.vector('float')()
   """
   # --- safer open mode: create if not exist, else append ---
   outpath = os.path.join(outdir, outname)
   newfile = ROOT.TFile.Open(outpath, "UPDATE")

   # --- determine branch names once ---
   if logMode:
       dr_branch_name = "log_1_over_deltaR"
   else:
       dr_branch_name = "deltaR"
   if mode == "z":
       val_branch_name = "log_1_over_z" if logMode else "z"
   else:
       val_branch_name = "log_kt" if logMode else "kt"

   # --- define local branch buffers (per-call) ---
   deltaR_vec = ROOT.std.vector('float')()
   val_vec = ROOT.std.vector('float')()

   # --- get or create the output tree ---
   # Always create a new TTree, let ROOT auto-index as ;1, ;2, ...
   lundTree = ROOT.TTree("lundTree", "Jet declustering kt and deltaR")

   # --- Determine dynamic branch names ---
   if logMode:
       dr_branch_name = "log_1_over_deltaR"
   else:
       dr_branch_name = "deltaR"

   if mode == "z":
       val_branch_name = "log_z" if logMode else "z"
   else:
       val_branch_name = "log_kt" if logMode else "kt"
   # --- Swap branch assignment if requested ---
   if swapAxes:
       lundTree.Branch(val_branch_name, deltaR_vec)
       lundTree.Branch(dr_branch_name, val_vec)
   else:
       lundTree.Branch(dr_branch_name, deltaR_vec)
       lundTree.Branch(val_branch_name, val_vec)



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
      # Compute according to user options

       if (lundPlane[k].delta_R > 0 and lundPlane[k].z > 0):
        # Compute according to user options
         if logMode:
             dr_val = math.log(1.0 / lundPlane[k].delta_R)
             if mode == "z":
                val = math.log(1.0 / lundPlane[k].z)
             else:
                val = math.log(lundPlane[k].kt)
         else:
             dr_val = lundPlane[k].delta_R
             if mode == "z":
                val = lundPlane[k].z
             else:
                val = lundPlane[k].kt

         # Push values to match branch name order
         if swapAxes:
             deltaR_vec.push_back(val)
             val_vec.push_back(dr_val)
         else:
             deltaR_vec.push_back(dr_val)
             val_vec.push_back(val)

     if len(val_vec) > 0:
        lundTree.Fill()
        val_vec.clear()
        deltaR_vec.clear()


   # Write the tree to the output file
   newfile.cd()
   lundTree.Write()
   newfile.Close();


import argparse
parser = argparse.ArgumentParser(description='Process benchmarks.')
parser.add_argument("--filename", help="", default="fileList.txt")
parser.add_argument("--treename", help="", default="tree")
parser.add_argument("--logMode", action="store_true", help="If set, output log(kt or z) and log(1/deltaR)")
parser.add_argument("--swapAxes", action="store_true", help="If set, swap the order of kt and deltaR in output")
parser.add_argument("--mode", type=str, choices=["kt", "z"], default="kt",
                    help="Select variable to pair with deltaR: 'kt' (default) or 'z'")



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
    '''
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
    loopFile("ignored.root", tree, logMode=opt.logMode, swapAxes=opt.swapAxes, mode=opt.mode);
    '''
    try:
        file = ROOT.TFile(line)
        # --- NEW: automatically iterate through all TTrees in the file ---
        keys = file.GetListOfKeys()
        for key in keys:
            obj = key.ReadObj()
            # Only process objects that are TTrees
            if isinstance(obj, ROOT.TTree):
                print(f"Processing tree: {obj.GetName()} in {line}")
                loopFile("ignored.root", obj, logMode=opt.logMode, swapAxes=opt.swapAxes, mode=opt.mode)
    except Exception as e:
        print(f"Failed to read file {line}: {e}")
        continue



    file.Close();
