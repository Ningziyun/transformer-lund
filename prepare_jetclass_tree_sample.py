# Convert ROOT jet constituents into full binary Lund-tree HDF5 files.

import argparse
import json
import os

import awkward as ak
import h5py
import numpy as np
import uproot

from ljpHelpers import constituents_to_lund_tree
from lund_tree_encoding import SPLIT_FEATURE_NAMES

N_FEATURES = len(SPLIT_FEATURE_NAMES)


class SplitWriter:
    def __init__(self, output_dir, split_name, max_nodes, source_files, max_constituents, jet_pt_min, radius):
        self.path = os.path.join(output_dir, f"{split_name}_fulltree.h5")
        self.file = h5py.File(self.path, "w")
        self.group = self.file.create_group("lund_tree")
        chunk_events = 128
        self.features = self.group.create_dataset(
            "features",
            shape=(0, max_nodes, N_FEATURES),
            maxshape=(None, max_nodes, N_FEATURES),
            chunks=(chunk_events, max_nodes, N_FEATURES),
            dtype="float32",
            compression="gzip",
            shuffle=True,
        )
        self.split = self.group.create_dataset(
            "split",
            shape=(0, max_nodes),
            maxshape=(None, max_nodes),
            chunks=(chunk_events, max_nodes),
            dtype="int8",
            compression="gzip",
            shuffle=True,
        )
        self.length = self.group.create_dataset(
            "length",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_events,),
            dtype="int32",
        )
        # Leading-jet four-vector, in the jet/{E,PX,PY,PZ} layout used by lunddata.py.
        jet_group = self.file.create_group("jet")
        self.jet = [
            jet_group.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(chunk_events,), dtype="float32")
            for name in ("E", "PX", "PY", "PZ")
        ]
        self.group.attrs["feature_order"] = ", ".join(SPLIT_FEATURE_NAMES)
        self.group.attrs["serialization"] = "depth-first preorder, harder child first"
        self.group.attrs["tree_constraint"] = "each node has either zero or two children"
        self.group.attrs["source_root_files"] = json.dumps([os.path.abspath(p) for p in source_files])
        self.group.attrs["max_constituents"] = int(max_constituents)
        self.group.attrs["jet_pt_min"] = float(jet_pt_min)
        self.group.attrs["radius"] = float(radius)
        self.rows = 0
        self.buffer_size = chunk_events
        self.feature_buffer = np.zeros((chunk_events, max_nodes, N_FEATURES), dtype=np.float32)
        self.split_buffer = np.full((chunk_events, max_nodes), -1, dtype=np.int8)
        self.length_buffer = np.zeros(chunk_events, dtype=np.int32)
        self.jet_buffer = np.zeros((chunk_events, 4), dtype=np.float32)
        self.buffered = 0

    def append(self, features, split, length, jet):
        self.feature_buffer[self.buffered] = features
        self.split_buffer[self.buffered] = split
        self.length_buffer[self.buffered] = length
        self.jet_buffer[self.buffered] = jet
        self.buffered += 1
        self.rows += 1
        if self.buffered == self.buffer_size:
            self.flush()

    def flush(self):
        if not self.buffered:
            return
        start = self.features.shape[0]
        stop = start + self.buffered
        self.features.resize(stop, axis=0)
        self.split.resize(stop, axis=0)
        self.length.resize(stop, axis=0)
        self.features[start:stop] = self.feature_buffer[: self.buffered]
        self.split[start:stop] = self.split_buffer[: self.buffered]
        self.length[start:stop] = self.length_buffer[: self.buffered]
        for component, dataset in enumerate(self.jet):
            dataset.resize(stop, axis=0)
            dataset[start:stop] = self.jet_buffer[: self.buffered, component]
        self.buffered = 0

    def close(self):
        self.flush()
        self.file.attrs["events"] = self.rows
        self.file.close()
        print(f"Saved {self.rows} {os.path.basename(self.path)} events: {self.path}", flush=True)


def build_full_lund_tree_files(
    root_files,
    output_dir,
    tree_name="tree",
    max_events=None,
    max_constituents=32,
    validation_fraction=0.1,
    test_fraction=0.1,
    seed=17,
    jet_pt_min=25.0,
    radius=1.0,
    step_size="128 MB",
):
    """Build train/validation/test tree HDF5 files from one or more ROOT files."""
    if not root_files:
        raise ValueError("Provide at least one ROOT input file")
    if max_constituents < 1:
        raise ValueError("max_constituents must be positive")
    if max_events is not None and max_events < 1:
        raise ValueError("max_events must be positive when provided")
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be non-negative and sum to less than one")

    os.makedirs(output_dir, exist_ok=True)
    max_nodes = 2 * max_constituents - 1
    rng = np.random.default_rng(seed)
    writers = {
        "train": SplitWriter(output_dir, "train", max_nodes, root_files, max_constituents, jet_pt_min, radius)
    }
    if validation_fraction > 0:
        writers["val"] = SplitWriter(
            output_dir, "val", max_nodes, root_files, max_constituents, jet_pt_min, radius
        )
    if test_fraction > 0:
        writers["test"] = SplitWriter(
            output_dir, "test", max_nodes, root_files, max_constituents, jet_pt_min, radius
        )

    probabilities = [1.0 - validation_fraction - test_fraction]
    labels = ["train"]
    if validation_fraction > 0:
        probabilities.append(validation_fraction)
        labels.append("val")
    if test_fraction > 0:
        probabilities.append(test_fraction)
        labels.append("test")

    total = 0
    try:
        for root_path in root_files:
            print(f"Reading {root_path}:{tree_name}", flush=True)
            with uproot.open(root_path) as source:
                if tree_name not in source:
                    raise KeyError(f"Tree {tree_name!r} not found in {root_path}")
                tree = source[tree_name]
                expressions = ["part_energy", "part_px", "part_py", "part_pz"]
                for arrays in tree.iterate(expressions, step_size=step_size, library="ak"):
                    batch_count = len(arrays["part_energy"])
                    for offset in range(batch_count):
                        if max_events is not None and total >= max_events:
                            break
                        vectors = np.column_stack(
                            [
                                ak.to_numpy(arrays["part_energy"][offset]),
                                ak.to_numpy(arrays["part_px"][offset]),
                                ak.to_numpy(arrays["part_py"][offset]),
                                ak.to_numpy(arrays["part_pz"][offset]),
                            ]
                        )[:max_constituents]
                        nodes, jet = constituents_to_lund_tree(
                            vectors, jet_pt_min=jet_pt_min, radius=radius, return_jet=True
                        )
                        if len(nodes) > max_nodes:
                            raise RuntimeError(
                                f"Event {total} has {len(nodes)} nodes; expected <= {max_nodes}"
                            )
                        node_features = np.zeros((max_nodes, N_FEATURES), dtype=np.float32)
                        node_splits = np.full(max_nodes, -1, dtype=np.int8)
                        node_features[: len(nodes)] = [node[0] for node in nodes]
                        node_splits[: len(nodes)] = [int(node[1]) for node in nodes]
                        selected = labels[int(rng.choice(len(labels), p=probabilities))]
                        writers[selected].append(node_features, node_splits, len(nodes), jet)
                        total += 1
                        if total % 1000 == 0:
                            counts = ", ".join(f"{name}={writer.rows}" for name, writer in writers.items())
                            print(f"Built {total} full trees ({counts})", flush=True)
                    if max_events is not None and total >= max_events:
                        break
            if max_events is not None and total >= max_events:
                break
    finally:
        for writer in writers.values():
            writer.close()

    if total == 0:
        raise RuntimeError("No events were processed from the provided ROOT files")
    print(f"Finished building {total} full Lund trees across {len(writers)} splits")
    return {name: writer.rows for name, writer in writers.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_files", nargs="+", help="ROOT files (shell globs are supported)")
    parser.add_argument("--tree-name", default="tree")
    parser.add_argument("--output-dir", default="inputFiles/jetclass_fulltree")
    parser.add_argument("--max-events", "--events", type=int, default=None, help="Optional cap across all input ROOT files")
    parser.add_argument("--max-constituents", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--jet-pt-min", type=float, default=25.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--step-size", default="128 MB", help="uproot chunk size")
    args = parser.parse_args()

    build_full_lund_tree_files(
        args.root_files,
        args.output_dir,
        tree_name=args.tree_name,
        max_events=args.max_events,
        max_constituents=args.max_constituents,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        jet_pt_min=args.jet_pt_min,
        radius=args.radius,
        step_size=args.step_size,
    )


if __name__ == "__main__":
    main()
