import argparse
import random
import time
import numpy as np
from . import logger
import torch
import os


class RunManager(object):
    def __init__(self):

        self.parser = argparse.ArgumentParser()

        # design inputs
        self.parser.add_argument("--pdb", type=str, default="pdbs/tim10.pdb", help="Input PDB")
        self.parser.add_argument(
            "--model_list",
            "--list",
            default=[
                "models/conditional_model_0.pt",
                "models/conditional_model_1.pt",
                "models/conditional_model_2.pt",
                "models/conditional_model_3.pt",
            ],
            nargs="+",
            help="Paths to conditional models",
        )
        self.parser.add_argument(
            "--init_model", type=str, default="models/baseline_model.pt", help="Path to baseline model (conditioned on backbone atoms only)"
        )

        # saving / logging
        self.parser.add_argument("--log_dir", type=str, default="./logs", help="Path to desired output log folder for designed structures")
        self.parser.add_argument("--seed", default=2, type=int, help="Random seed. Design runs are non-deterministic.")
        self.parser.add_argument("--save_rate", type=int, default=10, help="How often to save intermediate designed structures")

        # design parameters
        self.parser.add_argument(
            "--no_init_model", type=int, default=0, choices=(0, 1), help="Do not use baseline model to initialize sequence/rotmaers."
        )
        self.parser.add_argument(
            "--randomize",
            type=int,
            default=1,
            choices=(0, 1),
            help="Randomize starting sequence/rotamers for design. Toggle OFF to keep starting sequence and rotamers",
        )
        self.parser.add_argument(
            "--repack_only", type=int, default=0, choices=(0, 1), help="Only run rotamer repacking (no design, keep sequence fixed)"
        )
        self.parser.add_argument(
            "--use_rosetta_packer",
            type=int,
            default=0,
            choices=(0, 1),
            help="Use the Rosetta packer instead of the model for rotamer repacking during design",
        )
        self.parser.add_argument(
            "--threshold",
            type=float,
            default=20,
            help="Threshold in angstroms for defining conditionally independent residues for blocked sampling (should be greater than ~17.3)",
        )
        self.parser.add_argument("--symmetry", type=int, default=0, choices=(0, 1), help="Enforce symmetry during design")
        self.parser.add_argument(
            "--k", type=int, default=4, help="Enforce k-fold symmetry. Input pose length must be divisible by k. Requires --symmetry 1"
        )
        self.parser.add_argument("--ala", type=int, default=0, choices=(0, 1), help="Initialize sequence with poly-alanine")
        self.parser.add_argument("--val", type=int, default=0, choices=(0, 1), help="Initialize sequence with poly-valine")
        self.parser.add_argument("--restrict_gly", type=int, default=1, choices=(0, 1), help="Restrict no glycines for non-loop residues")
        self.parser.add_argument("--no_cys", type=int, default=0, choices=(0, 1), help="Enforce no cysteines in design")
        self.parser.add_argument("--no_met", type=int, default=0, choices=(0, 1), help="Enforce no methionines in design")
        self.parser.add_argument(
            "--pack_radius",
            type=float,
            default=5.0,
            help="Rosetta packer radius for rotamer packing after residue mutation. Must set --use_rosetta_packer 1.",
        )
        self.parser.add_argument(
            "--var_idx",
            type=str,
            default="",
            help="Path to txt file listing pose indices that should be designed/packed, all other side-chains will remain fixed. 0-indexed",
        )
        self.parser.add_argument(
            "--fixed_idx",
            type=str,
            default="",
            help="Path to txt file listing pose indices that should NOT be designed/packed, all other side-chains will be designed/packed. 0-indexed",
        )

        # optimization / sampling parameters
        self.parser.add_argument(
            "--anneal",
            type=int,
            default=1,
            choices=(0, 1),
            help="Option to do simulated annealing of average negative model pseudo-log-likelihood. Toggle OFF to do vanilla blocked sampling",
        )
        self.parser.add_argument("--step_rate", type=float, default=0.995, help="Multiplicative step rate for simulated annealing")
        self.parser.add_argument("--anneal_start_temp", type=float, default=1.0, help="Starting temperature for simulated annealing")
        self.parser.add_argument("--anneal_final_temp", type=float, default=0.0, help="Final temperature for simulated annealing")
        self.parser.add_argument("--n_iters", type=int, default=2500, help="Total number of iterations")

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self):
        self.args = self.parser.parse_args()

        self.log = logger.Logger(log_dir=self.args.log_dir)
        self.log.log_kvs(**self.args.__dict__)
        self.log.log_args(self.args)

        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        return self.args
