import ROOT
import fastjet
import math

# A structure to hold all of the information about the lund plane declustering
class Declustering:
  def __init__(self):
    # the (sub)jet, and its two declustering parts, for subsequent use
    self.jj = fastjet.PseudoJet()
    self.j1 = fastjet.PseudoJet()
    self.j2 = fastjet.PseudoJet()
    # variables of the (sub)jet about to be declustered
    self.pt = 0
    self.m = 0
    self.pt1 = 0
    self.pt2 = 0
    # properties of the declustering; NB kt is the _relative_ kt
    self.delta_R = 0
    self.z = 0
    self.kt = 0
    self.varphi = 0


# Code below from F. Dreyer
# https://github.com/rappoccio/fastjet-tutorial/blob/master/lund-jet-example/lund.hh
def jet_declusterings(jet_in):
  jd = fastjet.JetDefinition(fastjet.cambridge_algorithm, 1.0);
  rc = fastjet.Recluster(jd);
  j = rc.result(jet_in);

  result = [];
  jj = j;
  j1 = fastjet.PseudoJet()
  j2 = fastjet.PseudoJet()

  while (jj.has_parents(j1,j2)):
      declust = Declustering()

      # make sure j1 is always harder branch
      if (j1.pt2() < j2.pt2()):
        jTemp = j1;
        j1 = j2;
        j2 = jTemp;

      # store the subjets themselves
      declust.jj   = jj;
      declust.j1   = j1;
      declust.j2   = j2;

      # get info about the jet
      declust.pt   = jj.pt();
      declust.m    = jj.m();

      # collect info about the declustering
      declust.pt1     = j1.pt();
      declust.pt2     = j2.pt();
      declust.delta_R = j1.delta_R(j2);
      declust.z       = declust.pt2 / (declust.pt1 + declust.pt2);
      declust.kt      = j2.pt() * math.sin(declust.delta_R);

      # this is now phi along the jet axis, defined in a
      # long. boost. inv. way
      declust.varphi = math.atan2(j1.rap()-j2.rap(), j1.delta_phi_to(j2));

      # add it to our result
      result.append(declust);

      # follow harder branch
      jj = j1;

  return result;





