import fastjet
import math
import numpy as np

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

def jet_full_declusterings(jet_in, leaves=None):
  """Return a depth-first serialization of every node in a C/A reclustering.

  Each item is ``(features, splits)``. Internal nodes have the features in
  ``helpers_lundtree.SPLIT_FEATURE_NAMES``: ``[log(1/delta_R), log(z), varphi]`` for the split,
  with ``z = pt_soft / (pt_hard + pt_soft)``. Terminal leaves have zero features and
  ``splits=False``. The hard branch is visited before the soft branch,
  matching the ordering used by ``jet_declusterings``. With a ``leaves`` list,
  the leaf ``[E, px, py, pz]`` are appended to it in the same order.
  """
  jd = fastjet.JetDefinition(fastjet.cambridge_algorithm, 1.0)
  rc = fastjet.Recluster(jd)
  root = rc.result(jet_in)
  serialized = []

  def visit(node):
    j1 = fastjet.PseudoJet()
    j2 = fastjet.PseudoJet()
    if not node.has_parents(j1, j2):
      serialized.append(([0.0, 0.0, 0.0], False))
      if leaves is not None:
        leaves.append([node.e(), node.px(), node.py(), node.pz()])
      return

    if j1.pt2() < j2.pt2():
      j1, j2 = j2, j1

    delta_r = j1.delta_R(j2)
    z = j2.pt() / (j1.pt() + j2.pt())
    varphi = math.atan2(j1.rap() - j2.rap(), j1.delta_phi_to(j2))
    features = [
      math.log(1.0 / delta_r) if delta_r > 0 else 0.0,
      math.log(z) if z > 0 else 0.0,
      varphi,
    ]
    serialized.append((features, True))
    visit(j1)
    visit(j2)

  visit(root)
  return serialized


def constituents_to_lund_tree(vectors, jet_pt_min=25.0, radius=1.0, return_jet=False, leaves=None):
  """Return preorder nodes for the leading anti-kT jet, reclustered with C/A.

  ``vectors`` is an ``(N, 4)`` array ordered as ``[E, px, py, pz]``.
  Constituents are made massless (E = |p|) before clustering, matching the
  massless leaves of helpers_lundtree.reconstruct_tree, so a tree inverts exactly.
  Every internal node is followed by exactly two child subtrees; a leaf has
  no children. Each returned item is ``(features, split)``. With
  ``return_jet`` the jet ``[E, px, py, pz]`` (zeros without a jet) is also returned,
  and a ``leaves`` list collects the leaf four-vectors (see jet_full_declusterings).
  """
  empty = [([0.0, 0.0, 0.0], False)]
  no_jet = np.zeros(4)
  vectors = np.array(vectors, dtype=np.float64)
  vectors[:, 0] = np.linalg.norm(vectors[:, 1:], axis=1)
  valid = np.isfinite(vectors).all(axis=1) & (vectors[:, 0] > 0)
  vectors = vectors[valid]
  if len(vectors) == 0:
    return (empty, no_jet) if return_jet else empty

  particles = [
    fastjet.PseudoJet(float(px), float(py), float(pz), float(energy))
    for energy, px, py, pz in vectors
  ]
  jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, radius, fastjet.E_scheme)
  cluster_sequence = fastjet.ClusterSequence(particles, jet_def)
  jets = fastjet.sorted_by_pt(cluster_sequence.inclusive_jets(jet_pt_min))
  if not jets:
    return (empty, no_jet) if return_jet else empty

  # Keep the ClusterSequence alive until the C/A reclustering and traversal end.
  tree = jet_full_declusterings(jets[0], leaves)
  jet = np.array([jets[0].e(), jets[0].px(), jets[0].py(), jets[0].pz()])
  del cluster_sequence
  return (tree, jet) if return_jet else tree
