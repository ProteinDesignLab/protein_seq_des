import numpy as np
import os
import torch
import torch.nn as nn

from seq_des import *
import seq_des.sampler as sampler
import seq_des.models as models

import common.run_manager
import common.atoms

import sys
import pickle
import glob

from pyrosetta.rosetta.protocols.simple_filters import BuriedUnsatHbondFilterCreator, PackStatFilterCreator
from pyrosetta.rosetta.protocols.denovo_design.filters import ExposedHydrophobicsFilterCreator

from tqdm import tqdm

__author__ = 'Namrata Anand-Achim'


def log_metrics(run="sampler", args=None, log=None, iteration=0, design_sampler=None, prefix=""):
    # tensorboard logging

    # log structure / sequence metrics
    log.log_scalar("{run}/{prefix}rosetta_energy".format(run=run, prefix=prefix), design_sampler.rosetta_energy)
    log.log_scalar("{run}/{prefix}seq_overlap".format(run=run, prefix=prefix), design_sampler.seq_overlap)
    log.log_scalar("{run}/{prefix}anneal_start_temp".format(run=run, prefix=prefix), design_sampler.anneal_start_temp)
    log.log_scalar("{run}/{prefix}anneal_final_temp".format(run=run, prefix=prefix), design_sampler.anneal_final_temp)
    log.log_scalar("{run}/{prefix}log_p".format(run=run, prefix=prefix), design_sampler.log_p_mean.item())
    log.log_scalar("{run}/{prefix}chi_error".format(run=run, prefix=prefix), design_sampler.chi_error)
    log.log_scalar("{run}/{prefix}chi_rmsd".format(run=run, prefix=prefix), design_sampler.chi_rmsd)

    # log rosetta score terms
    for s in design_sampler.score_terms:
        log.log_scalar("{run}/z_{prefix}{s}".format(run=run, prefix=prefix, s=s), float(design_sampler.curr_score_terms[s].mean()))

    # log rosetta agnostic terms
    for n, s in design_sampler.filter_scores:
        log.log_scalar("{run}/y_{prefix}{n}".format(run=run, prefix=prefix, n=n), s)



def load_model(model, use_cuda=True, nic=len(common.atoms.atoms)):
    classifier = models.seqPred(nic=nic)
    if use_cuda:
        classifier.cuda()
    if use_cuda:
        state = torch.load(model)
    else:
        state = torch.load(model, map_location="cpu")
    for k in state.keys():
        if "module" in k:
            print("MODULE")
            classifier = nn.DataParallel(classifier)
        break
    if use_cuda:
        classifier.load_state_dict(torch.load(model))
    else:
        classifier.load_state_dict(torch.load(model, map_location="cpu"))
    return classifier


def load_models(model_list, use_cuda=True, nic=len(common.atoms.atoms)):
    classifiers = []
    for model in model_list:
        classifier = load_model(model, use_cuda=use_cuda, nic=nic)
        classifiers.append(classifier)
    return classifiers


def main():

    manager = common.run_manager.RunManager()

    manager.parse_args()
    args = manager.args
    log = manager.log

    use_cuda = torch.cuda.is_available()

    # download pdb if not there already
    if not os.path.isfile(args.pdb):
        print("Downloading pdb to current directory...")
        os.system("wget -O {} https://files.rcsb.org/download/{}.pdb".format(args.pdb, args.pdb[:-4].upper()))

    assert os.path.isfile(args.pdb), "pdb not found"

    # load models
    if args.init_model != "":
        init_classifier = load_model(args.init_model, use_cuda=use_cuda, nic=len(common.atoms.atoms))
        init_classifier.eval()
        init_classifiers = [init_classifier]
    else:
        assert not (args.ala and args.val), "must specify either poly-alanine or poly-valine"
        if args.randomize:
            if args.ala:
                init_scheme = "poly-alanine"
            elif args.val:
                init_scheme = "poly-valine"
            else:
                init_scheme = "random"
        else: init_scheme = 'using starting structure'
        print("No baseline model specified, initialization will be %s" % init_scheme)
        init_classifiers = None

    classifiers = load_models(args.model_list, use_cuda=use_cuda, nic=len(common.atoms.atoms) + 1 + 21)
    for classifier in classifiers:
        classifier.eval()

    # set up design_sampler
    design_sampler = sampler.Sampler(args, classifiers, init_classifiers, log=log, use_cuda=use_cuda)

    # initialize sampler
    design_sampler.init()

    # log metrics for gt seq/structure
    log_metrics(args=args, log=log, iteration=0, design_sampler=design_sampler, prefix="gt_")
    best_rosetta_energy = np.inf
    best_energy = np.inf

    # initialize design_sampler sequence with baseline model prediction or random/poly-alanine/poly-valine initial sequence, save initial model
    design_sampler.init_seq()
    design_sampler.pose.dump_pdb(log.log_path + "/" + "curr_0_%s.pdb" % (log.ts))

    # run design
    with torch.no_grad():
        for i in tqdm(range(1, int(args.n_iters)), desc='running design'):

            # step
            design_sampler.step()

            # logging
            log_metrics(args=args, log=log, iteration=i, design_sampler=design_sampler)

            if design_sampler.log_p_mean < best_energy:
                design_sampler.pose.dump_pdb(log.log_path + "/" + "curr_best_log_p_%s.pdb" % log.ts)
                best_energy = design_sampler.log_p_mean

            if design_sampler.rosetta_energy < best_rosetta_energy:
                design_sampler.pose.dump_pdb(log.log_path + "/" + "curr_best_rosetta_energy_%s.pdb" % log.ts)
                best_rosetta_energy = design_sampler.rosetta_energy

            # save intermediate models -- comment out if desired
            if (i==1) or (i % args.save_rate == 0) or (i == args.n_iters - 1):
                design_sampler.pose.dump_pdb(log.log_path + "/" + "curr_%s_%s.pdb" % (i, log.ts))

            log.advance_iteration()

    # save final model
    design_sampler.pose.dump_pdb(log.log_path + "/" + "curr_final.pdb")


if __name__ == "__main__":
    main()
