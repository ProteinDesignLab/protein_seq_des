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

        self.parser.add_argument("--workers", type=int, help="number of data loading workers", default=0)
        self.parser.add_argument("--cuda", type=int, default=1, help="enables cuda")

        # training parameters
        self.parser.add_argument("--batchSize", type=int, default=64, help="input batch size")
        self.parser.add_argument("--ngpu", type=int, default=1, help="num gpus to parallelize over")

        self.parser.add_argument("--nf", type=int, default=64, help="base number of filters")
        self.parser.add_argument("--txt", type=str, default="data/train_domains_s95.txt", help="default txt input file")

        self.parser.add_argument("--epochs", type=int, default=100, help="enables cuda")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--reg", type=float, default=5e-6, help="L2 regularization")
        self.parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5")
        self.parser.add_argument("--momentum", type=float, default=0.01, help="momentum for batch norm")

        self.parser.add_argument(
            "--model", type=str, default="", help="path to saved pretrained model for resuming training",
        )
        self.parser.add_argument("--optimizer", type=str, default="", help="path to saved optimizer params")
        self.parser.add_argument(
            "--validation_frequency", type=int, default=500, help="how often to validate during training",
        )
        self.parser.add_argument("--save_frequency", type=int, default=2000, help="how often to save models")
        self.parser.add_argument("--sync_frequency", type=int, default=1000, help="how often to sync to GCP")

        self.parser.add_argument(
            "--num_return", type=int, default=400, help="number of nearest non-side-chain atmos to return per voxel",
        )
        self.parser.add_argument("--chunk_size", type=int, default=10000, help="chunk size for saved coordinate tensors")

        self.parser.add_argument("--data_dir", type=str, default="/data/simdev_2tb/protein/sequence_design/data/coords")
        self.parser.add_argument("--pdb_dir", type=str, default="/data/drive2tb/protein/pdb")
        self.parser.add_argument("--save_dir", type=str, default="./coords")

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
            "--init_model", type=str, default="models/baseline_model.pt", help="Path to baseline model (conditioned on backbone atoms only)",
        )

        # saving / logging
        self.parser.add_argument(
            "--log_dir", type=str, default="./logs", help="Path to desired output log folder for designed structures",
        )
        self.parser.add_argument("--seed", default=2, type=int, help="Random seed. Design runs are non-deterministic.")
        self.parser.add_argument(
            "--save_rate", type=int, default=10, help="How often to save intermediate designed structures",
        )

        # design parameters
        self.parser.add_argument(
            "--no_init_model", type=int, default=0, choices=(0, 1), help="Do not use baseline model to initialize sequence/rotmaers.",
        )
        self.parser.add_argument(
            "--randomize",
            type=int,
            default=1,
            choices=(0, 1),
            help="Randomize starting sequence/rotamers for design. Toggle OFF to keep starting sequence and rotamers",
        )
        self.parser.add_argument(
            "--repack_only", type=int, default=0, choices=(0, 1), help="Only run rotamer repacking (no design, keep sequence fixed)",
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
            "--k", type=int, default=4, help="Enforce k-fold symmetry. Input pose length must be divisible by k. Requires --symmetry 1",
        )
        self.parser.add_argument(
            "--ala", type=int, default=0, choices=(0, 1), help="Initialize sequence with poly-alanine",
        )
        self.parser.add_argument(
            "--val", type=int, default=0, choices=(0, 1), help="Initialize sequence with poly-valine",
        )
        self.parser.add_argument(
            "--restrict_gly", type=int, default=1, choices=(0, 1), help="Restrict no glycines for non-loop residues",
        )
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
        self.parser.add_argument("--do_mcmc", type=int, default=0, help="Option to do Metropolis-Hastings")
        self.parser.add_argument(
            "--step_rate", type=float, default=0.995, help="Multiplicative step rate for simulated annealing",
        )
        self.parser.add_argument(
            "--anneal_start_temp", type=float, default=1.0, help="Starting temperature for simulated annealing",
        )
        self.parser.add_argument(
            "--anneal_final_temp", type=float, default=0.0, help="Final temperature for simulated annealing",
        )
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
