import math
import numpy as np
from collections import namedtuple

'''
Everything about the full Lund-tree representation, apart from the fastjet clustering (ljpHelpers.py):
  - the token layout the models see, and the encoding of split features into an unbounded space
  - walking a tree node-by-node (used for the encoding, the model generation and the plots)
  - converting tokens back into trees and reconstructing the constituent 4-vectors

A jet is its full C/A declustering tree, stored as depth-first preorder nodes with the harder child first.
Every node is either a split, with features [log(1/delta_R), log(z), varphi], or a leaf with zero features.
The jet pT, eta and phi are the conditioning, the jet mass and all constituents then follow from the splits.

Model tokens are [Nnodes, Nfeat+Ncontext+1] (see TOKEN_NAMES):
  preprocess split features | input-only context of the next node | split flag (1=split, 0=leaf, -1=pad)
Physical tokens are [Nnodes, Nfeat+1]: unprcoessed split features | split flag
'''

# ---------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------
SPLIT_FEATURE_NAMES = ("log_inv_delta_R", "log_z", "varphi")  #features of a split, relative to its parent node
CONTEXT_NAMES = ("log1p_depth", "is_soft_child", "log_pt_fraction")  #input-only context of the next node
TOKEN_NAMES = SPLIT_FEATURE_NAMES + CONTEXT_NAMES + ("split",)  #model token columns, also the lunddata.py dataset names

#Column of each feature in node features and tokens
DR, Z, VARPHI = range(len(SPLIT_FEATURE_NAMES))

#Plot labels of each physical node feature, in column order
FEATURE_LABELS = [r"$\log(1/\Delta R)$", r"$\log(z)$", r"$\varphi$"]

#Physical bounds on the split features. The model learns and samples them in an unbounded space, so every generated split
#decodes to a kinematically valid one.
DELTA_R_MAX = 1.0  #C/A reclustering radius in ljpHelpers.jet_full_declusterings (a bit above the R=0.8 jets, for wiggle room)
Z_MAX = 0.5  #the softer branch never carries more than half the scalar pT

#C/A angular ordering: with E-scheme recombination the merged axis lies between the two merged objects, so the next merge can
#be up to sqrt(3)/2 closer than the last one. A split's delta_R is bounded by DELTA_R_RATIO_MAX times its parent's delta_R.
DELTA_R_RATIO_MAX = 2.0 / np.sqrt(3.0)

# ---------------------------------------------------------------------
# Split kinematics and the model-space encoding
# ---------------------------------------------------------------------
def pt_sharing(delta_r, z, varphi):
    '''
    Return (pt_hard + pt_soft) / pt_parent of a split.
    The daughter pT vectors sum to the parent pT and are delta_phi = -delta_R*cos(varphi) apart in azimuth
    (varphi = atan2(y_hard - y_soft, phi_soft - phi_hard)), with z the soft fraction of their scalar sum.
    Then pt_soft = z * sharing * pt_parent and kt = pt_soft * sin(delta_R).
    '''
    delta_phi = -delta_r * np.cos(varphi)
    return 1.0 / np.sqrt((1 - z) ** 2 + z**2 + 2 * z * (1 - z) * np.cos(delta_phi))

def preprocess_features(features, delta_r_max=DELTA_R_MAX):
    '''
    Map physical split features to the model's unbounded space:
    log(1/delta_R) -> softplus^-1(log(delta_r_max/delta_R)),  log(z) -> softplus^-1(log(Z_MAX/z)),  varphi unchanged.
    delta_r_max is a scalar or a per-row array (see preprocess_trees).
    '''
    prepocessed = np.array(features, dtype=np.float64)
    for column, gap in ((DR, prepocessed[..., DR] + np.log(delta_r_max)), (Z, np.log(Z_MAX) - prepocessed[..., Z])):
        gap = np.maximum(gap, 1e-6)
        prepocessed[..., column] = gap + np.log(-np.expm1(-gap))
    return prepocessed

def undo_preprocess_features(prepocessed, delta_r_max=DELTA_R_MAX):
    '''Inverse of preprocess_features, any real input un-processes inside the physical bounds'''
    features = np.array(prepocessed, dtype=np.float64)
    features[..., DR] = np.logaddexp(0.0, features[..., DR]) - np.log(delta_r_max)
    features[..., Z] = np.log(Z_MAX) - np.logaddexp(0.0, features[..., Z])
    return features

# ---------------------------------------------------------------------
# Walking a tree node by node
# ---------------------------------------------------------------------
#A node met while walking: its depth, whether it is the soft child, its pT, whether it lies on the primary Lund plane
#(the hard branch from the root) and its delta_R bound
TreeNode = namedtuple("TreeNode", ["depth", "is_soft", "pt", "primary", "delta_r_bound"])

#The physical description of a split
Split = namedtuple("Split", ["delta_r", "z", "varphi"])

class TreeWalker:
    '''
    Follows a depth-first preorder tree node by node, keeping a stack of the open child slots.
    step() consumes the next node and returns it as a TreeNode plus, for a split, its Split. Node pTs follow down the tree
    from root_pt through pt_sharing. With prepocessed=True the features are in model space, and each delta_R is reverse preprocessed with
    its parent's bound (DELTA_R_RATIO_MAX times the parent delta_R, the root uses DELTA_R_MAX).
    context() gives the context (CONTEXT_NAMES) of the next node, with pTs relative to root_pt.
    '''
    def __init__(self, root_pt=1.0, prepocessed=False):
        self.prepocessed = prepocessed
        self.stack = [TreeNode(0, False, float(root_pt), True, DELTA_R_MAX)]  #open child slots, next on top

    def next_node(self):
        #The node the next step consumes, or None once the tree is complete
        return self.stack[-1] if self.stack else None

    def context(self):
        #The context of the next node, or zeros once the tree is complete
        node = self.next_node()
        if node is None:
            return np.zeros(len(CONTEXT_NAMES))
        return np.array([np.log1p(node.depth), float(node.is_soft), np.log(node.pt)])

    def step(self, features, split):
        node = self.stack.pop()
        if not split:
            return node, None
        physical = undo_preprocess_features(features, node.delta_r_bound) if self.prepocessed else features
        delta_r, z, varphi = np.exp(-physical[DR]), np.exp(physical[Z]), physical[VARPHI]
        scalar_pt = node.pt * pt_sharing(delta_r, z, varphi)
        child_bound = min(DELTA_R_MAX, DELTA_R_RATIO_MAX * delta_r)
        self.stack.append(TreeNode(node.depth + 1, True, z * scalar_pt, False, child_bound))  #soft child, visited second
        self.stack.append(TreeNode(node.depth + 1, False, (1 - z) * scalar_pt, node.primary, child_bound))  #hard child, next
        return node, Split(delta_r, z, varphi)

def walk_tree(tokens, root_pt=1.0, prepocessed=True):
    '''Yield (index, node, split) for every node of one [Nnodes, F+1] token sequence (see TreeWalker)'''
    walker = TreeWalker(root_pt, prepocessed)
    for index, token in enumerate(tokens):
        if walker.next_node() is None or token[-1] not in (0, 1):
            return
        node, split = walker.step(token[:-1], token[-1] == 1)
        yield index, node, split

# ---------------------------------------------------------------------
# Physical <-> model tokens
# ---------------------------------------------------------------------
def preprocess_trees(features, splits):
    '''
    preprocess physical [N, Nnodes, F] trees with [N, Nnodes] split flags, in one walk over each tree.
    Returns the model-space features (each split's delta_R bounded relative to its parent's, see TreeWalker) and the
    [N, Nnodes, len(CONTEXT_NAMES)] context of node k+1 at row k (zeros past the end).
    '''
    features = np.array(features, dtype=np.float64)
    bounds = np.full(splits.shape, DELTA_R_MAX)
    context = np.zeros(features.shape[:2] + (len(CONTEXT_NAMES),))
    for jet in range(len(features)):
        walker = TreeWalker()
        for index, split in enumerate(splits[jet]):
            if walker.next_node() is None or split not in (0, 1):
                break
            bounds[jet, index] = walker.next_node().delta_r_bound
            walker.step(features[jet, index], split == 1)
            context[jet, index] = walker.context()

    prepocessed = np.array(features, dtype=np.float64)
    internal = splits == 1
    prepocessed[internal] = preprocess_features(features[internal], bounds[internal])
    return prepocessed, context

def physical_tokens(tokens):
    '''
    Reverse preprocess of model tokens [N, Nnodes, F+Ncontext+1] to physical tokens [N, Nnodes, F+1] for plotting and reconstruction.
    Drops the context columns and decodes each split with its parent's delta_R bound. Leaves keep zero features, padding
    keeps the pad value.
    '''
    tokens = np.concatenate([tokens[..., :len(SPLIT_FEATURE_NAMES)], tokens[..., -1:]], axis=-1).astype(np.float64)
    for jet in range(len(tokens)):
        for index, _, split in walk_tree(tokens[jet]):
            if split is not None:
                tokens[jet, index, [DR, Z, VARPHI]] = [-np.log(split.delta_r), np.log(split.z), split.varphi]
    return tokens

def nodes_to_tree(nodes):
    '''Nest preorder (features, split) nodes into {"features": [...], "children": [hard, soft]} dictionaries'''
    cursor = iter(nodes)

    def build():
        features, split = next(cursor)
        return {"features": [float(value) for value in features], "children": [build(), build()] if split else []}

    return build()

def tokens_to_tree(tokens):
    '''Nest one physical [Nnodes, F+1] token sequence, or None when it is not a complete binary tree'''
    nodes = []
    pending = 1
    for row in np.asarray(tokens, dtype=np.float64):
        if not pending:
            break
        split = row[-1] == 1
        nodes.append((row[:-1], split))
        pending += 1 if split else -1
    return None if pending else nodes_to_tree(nodes)

# ---------------------------------------------------------------------
# Reconstruct the constituent 4-vectors from a tree
# ---------------------------------------------------------------------
def _split_transverse(parent_pt, features):
    '''Invert the transverse part of a split: (pt_hard, pt_soft, delta_y, delta_phi), independent of rapidity'''
    log_inv_dr, log_z, varphi = (float(value) for value in features[:3])
    if not all(math.isfinite(value) for value in (log_inv_dr, log_z, varphi)):
        raise ValueError("Split features must all be finite")

    delta_r = math.exp(-log_inv_dr)
    z = math.exp(log_z)
    if not 0 < z < 1:
        raise ValueError(f"z={z:g} is outside (0, 1)")

    #jet_full_declusterings uses varphi = atan2(y_hard - y_soft, hard.delta_phi_to(soft)),
    #and fastjet's delta_phi_to returns phi_soft - phi_hard, so negate to get hard - soft
    angle = math.remainder(varphi, 2.0 * math.pi)
    delta_y = delta_r * math.sin(angle)
    delta_phi = -delta_r * math.cos(angle)

    #z = pt_soft / (pt_hard + pt_soft), and the parent pT is the vector sum of the daughter pTs, which fixes the scalar sum
    with np.errstate(divide="ignore"):
        scalar_pt = parent_pt * float(pt_sharing(delta_r, z, angle))
    if not math.isfinite(scalar_pt):
        raise ValueError("Back-to-back equal-pT split cannot carry the parent pT")
    return (1 - z) * scalar_pt, z * scalar_pt, delta_y, delta_phi

def _subtree_masses(node, parent_pt, masses):
    '''Return a subtree's invariant mass from its parent pT, storing every node's mass in masses by id'''
    children = node["children"]
    mass = 0.0
    if children:
        pt_hard, pt_soft, delta_y, delta_phi = _split_transverse(parent_pt, node["features"])
        mt_hard = math.hypot(pt_hard, _subtree_masses(children[0], pt_hard, masses))
        mt_soft = math.hypot(pt_soft, _subtree_masses(children[1], pt_soft, masses))
        energy = mt_hard * math.cosh(delta_y) + mt_soft
        px = pt_hard * math.cos(delta_phi) + pt_soft
        py = pt_hard * math.sin(delta_phi)
        pz = mt_hard * math.sinh(delta_y)
        mass = math.sqrt(max(energy**2 - px**2 - py**2 - pz**2, 0.0))
    masses[id(node)] = mass
    return mass

def _split_parent(parent, features, mass_hard, mass_soft):
    '''Invert one split of the parent [E, px, py, pz] into the two (possibly massive) daughter 4-vectors'''
    energy, px, py, pz = (float(value) for value in parent)
    parent_pt = math.hypot(px, py)
    if parent_pt <= 0 or energy <= abs(pz):
        raise ValueError("Parent must have positive transverse momentum and finite rapidity")
    pt_hard, pt_soft, delta_y, delta_phi = _split_transverse(parent_pt, features)
    mt_hard = math.hypot(pt_hard, mass_hard)
    mt_soft = math.hypot(pt_soft, mass_soft)

    #Rapidities so the daughter light-cone momenta add up to the parent's
    plus_factor = mt_hard * math.exp(delta_y) + mt_soft
    minus_factor = mt_hard * math.exp(-delta_y) + mt_soft
    rapidity_soft = 0.5 * (math.log((energy + pz) / (energy - pz)) - math.log(plus_factor / minus_factor))
    rapidity_hard = rapidity_soft + delta_y

    #Orient the daughter azimuths so their vector sum points along the parent pT
    phi_soft = math.atan2(py, px) + math.atan2(-pt_hard * math.sin(delta_phi), pt_hard * math.cos(delta_phi) + pt_soft)
    phi_hard = phi_soft + delta_phi

    def branch_p4(pt, mt, rapidity, phi):
        return np.asarray([mt * math.cosh(rapidity), pt * math.cos(phi), pt * math.sin(phi), mt * math.sinh(rapidity)], dtype=np.float64)

    hard = branch_p4(pt_hard, mt_hard, rapidity_hard, phi_hard)
    soft = branch_p4(pt_soft, mt_soft, rapidity_soft, phi_soft)

    #Closure check
    residual = hard + soft - np.asarray(parent, dtype=np.float64)
    scale = max(float(np.max(np.abs(parent))), 1.0)
    if np.max(np.abs(residual)) > 2e-6 * scale:
        raise ValueError(f"Split coordinates are inconsistent with the supplied parent four-vector (maximum closure residual {np.max(np.abs(residual)) / scale:.3g})")
    return hard, soft

def root_four_vector(tree, root_kinematics, masses=None):
    '''
    Return the jet [E, px, py, pz] from root_kinematics=(pt, eta, phi), the mass follows from the splits.
    masses collects every subtree mass by node id (see _subtree_masses).
    '''
    if not tree["children"]:
        raise ValueError("A tree without splits carries no jet kinematics")
    pt, eta, phi = (float(value) for value in root_kinematics)
    if not (all(math.isfinite(value) for value in (pt, eta, phi)) and pt > 0):
        raise ValueError("Root kinematics must all be finite with a positive pT")
    mt = math.hypot(pt, _subtree_masses(tree, pt, {} if masses is None else masses))
    rapidity = math.asinh(pt * math.sinh(eta) / mt)  #pz = pt sinh(eta) = mt sinh(rapidity)
    return np.asarray([mt * math.cosh(rapidity), pt * math.cos(phi), pt * math.sin(phi), mt * math.sinh(rapidity)], dtype=np.float64)

def reconstruct_tree(tree, root_kinematics):
    '''
    Return the jet 4-vector and its leaf constituents (massless, depth-first hard-first order) from a nested tree and
    the jet (pt, eta, phi). The constituents always sum to the jet.
    '''
    masses = {}
    root_p4 = root_four_vector(tree, root_kinematics, masses)

    def visit(node, parent):
        children = node["children"]
        if not children:
            return [np.asarray(parent, dtype=np.float64)]
        hard, soft = _split_parent(parent, node["features"], masses[id(children[0])], masses[id(children[1])])
        return visit(children[0], hard) + visit(children[1], soft)

    constituents = visit(tree, root_p4)
    total = np.sum(constituents, axis=0)
    scale = max(float(np.max(np.abs(root_p4))), 1.0)
    if np.max(np.abs(total - root_p4)) > 1e-5 * scale:
        raise ValueError("Reconstructed constituents do not conserve the root four-vector")
    return root_p4, constituents

def tokens_to_constituents(tokens, n_constituents, root_kinematics, pad_value=-1):
    '''
    Reconstruct [N, n_constituents, 4] constituent [E, px, py, pz] from physical [N, Nnodes, F+1] tokens and the
    [N, 3] jet (pt, eta, phi). Unused slots, and every slot of a jet that cannot be reconstructed, are pad_value.
    Returns the constituents and the number of failed jets.
    '''
    constituents = np.full((len(tokens), n_constituents, 4), float(pad_value))
    failures = 0
    for index, jet_tokens in enumerate(tokens):
        tree = tokens_to_tree(jet_tokens)
        try:
            if tree is None:
                raise ValueError("incomplete tree")
            _, leaves = reconstruct_tree(tree, root_kinematics[index])
        except ValueError:
            failures += 1
            continue
        leaves = np.asarray(leaves)[:n_constituents]
        constituents[index, : len(leaves)] = leaves
    return constituents, failures

# ---------------------------------------------------------------------
# Summaries for the plots
# ---------------------------------------------------------------------
def lund_plane_points(tokens, root_pt):
    '''Return [log(kt), log(1/delta_R)] of every split on the primary Lund plane, from physical tokens and the [N] jet pTs'''
    points = []
    for row, jet_pt in zip(tokens, np.asarray(root_pt, dtype=np.float64)):
        for _, node, split in walk_tree(row, jet_pt, prepocessed=False):
            if split is not None and node.primary:
                kt = split.z * node.pt * pt_sharing(*split) * np.sin(split.delta_r)
                points.append((np.log(kt), -np.log(split.delta_r)))
    return np.array(points).reshape(-1, 2)

def summarize_trees(tokens, root_kinematics, reconstruct=True):
    '''
    Describe physical [N, Nnodes, F+1] tokens of jets with [N, 3] root_kinematics (pt, eta, phi). Returns a dict of
      tokens         the physical tokens
      split          [N, Nnodes] split flags (1 split, 0 leaf, -1 padding)
      features       [N, Nnodes, F] physical split features, NaN for non-splits
      primary        [M, 2] primary Lund-plane points, see lund_plane_points
      constituents   [N, Nnodes//2+1, 4] reconstructed [E, px, py, pz], padded with -1 (with reconstruct)
      failed         the number of jets that could not be reconstructed (with reconstruct)
    '''
    split = tokens[..., -1]
    summary = {
        "tokens": tokens,
        "split": split,
        "features": np.where((split == 1)[..., None], tokens[..., :-1], np.nan),
        "primary": lund_plane_points(tokens, root_kinematics[:, 0]),
    }
    if reconstruct:
        summary["constituents"], summary["failed"] = tokens_to_constituents(tokens, tokens.shape[1] // 2 + 1, root_kinematics)
    return summary

def load_lund_jets(path, reconstruct=True):
    '''
    Load a samples file (validation_samples.npz or generate_lund_trees.py output) into {label: sample} for the original
    jets and each generated set, where every sample is the summarize_trees dict plus "jets", the [N, 3] conditioning.
    '''
    data = np.load(path, allow_pickle=True)
    jets = data["jet_list"]
    root = np.asarray(jets, dtype=np.float64)[:, :3]
    labels = data["labels"].tolist()

    arrays = {labels[0]: data["original"]}
    for i, label in enumerate(labels[1:]):
        if f"generated_{i}" in data:
            arrays[label] = data[f"generated_{i}"]

    return {label: {**summarize_trees(physical_tokens(tokens), root, reconstruct), "jets": jets} for label, tokens in arrays.items()}
