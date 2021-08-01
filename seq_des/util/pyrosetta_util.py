import numpy as np
import common.atoms

from rosetta import *
from pyrosetta import *
init("-mute basic -mute core -mute protocols  -ex1 -ex2 -constant_seed")

#from pyrosetta.toolbox import pose_from_rcsb, cleanATOM  # , mutate_residue
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

from pyrosetta.rosetta.core import conformation
from pyrosetta.rosetta.core import chemical
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

score_manager = pyrosetta.rosetta.core.scoring.ScoreTypeManager()
scorefxn = get_fa_scorefxn()
from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code


def get_seq_delta(s1, s2):
    count = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            count += 1
    return count


def score_pose(pose):
    return scorefxn(pose)


def randomize_sequence(new_seq, pose, pack_radius=5.0, fixed_idx=[], var_idx=[], ala=False, val=False, resfile=False, enforce=False, repack_rotamers=0):
    for idx in range(pose.residues.__len__()):
        # do not mutate fixed indices / only mutate var indices
        if idx in fixed_idx:
            continue
        elif len(var_idx) > 0 and idx not in var_idx:
            continue

        res = pose.residue(idx + 1)
        ref_res_name = res.name()

        if ":" in ref_res_name:
            ref_res_name = ref_res_name[: ref_res_name.find(":")]
        if "_" in ref_res_name:
            ref_res_name = ref_res_name[: ref_res_name.find("_")]

        if ref_res_name not in common.atoms.res_label_dict.keys():
            continue

        if ala:
            r = common.atoms.res_label_dict["ALA"]
        elif val:
            r = common.atoms.res_label_dict["VAL"]
        else:
            r = new_seq[idx]

        res_aa = common.atoms.aa_map[r]
        
        # resfile hangling: ex. 5 PIKAA C means set the initial sequence at residue 5 to 'C'
        if idx in resfile.keys():
            res_aa = resfile[idx]

        pose = handle_disulfide(pose, idx)
        mutate_residue(pose, idx + 1, res_aa, pack_radius=pack_radius, repack_rotamers=repack_rotamers)

    return pose, pose.residues.__len__()


# from https://github.com/barricklab/mutant-protein-stability/blob/master/PyRosetta_TACC_MPI.py
def handle_disulfide(pose, idx):
    res = pose.residue(idx + 1)
    if (res.name() == "CYS:disulfide") or (res.name() == "CYD"):
        disulfide_partner = None
        try:
            disulfide_partner = res.residue_connection_partner(res.n_residue_connections())
        except AttributeError:
            disulfide_partner = res.residue_connection_partner(res.n_current_residue_connections())
        temp_pose = pyrosetta.Pose()
        temp_pose.assign(pose)
        # (Packing causes seg fault if current CYS residue is not
        # also converted before mutating.)
        conformation.change_cys_state(idx + 1, "CYS", temp_pose.conformation())
        conformation.change_cys_state(disulfide_partner, "CYS", temp_pose.conformation())
        pose = temp_pose
    return pose


def mutate(pose, idx, res, pack_radius=5.0, fixed_idx=[], var_idx=[], repack_rotamers=0):
    if idx in fixed_idx:
        return pose
    elif len(var_idx) > 0 and idx not in var_idx:
        return pose
    pose = handle_disulfide(pose, idx)
    pose = mutate_residue(pose, idx + 1, res, pack_radius=pack_radius, repack_rotamers=repack_rotamers)
    return pose


def mutate_list(pose, idx_list, res_list, pack_radius=5.0, fixed_idx=[], var_idx=[], repack_rotamers=0):
    assert len(idx_list) == len(res_list), (len(idx_list), len(res_list))
    for i in range(len(idx_list)):
        idx, res = idx_list[i], res_list[i]
        if len(fixed_idx) > 0 and idx in fixed_idx:
            continue
        if len(var_idx) > 0 and idx not in var_idx:
            continue
        sequence = pose.sequence()
        pose = mutate(pose, idx, res, pack_radius=pack_radius, fixed_idx=fixed_idx, var_idx=var_idx, repack_rotamers=repack_rotamers)
        new_sequence = pose.sequence()
        assert get_seq_delta(sequence, new_sequence) <= 1, get_seq_delta(sequence, new_sequence)
        assert res == pose.sequence()[idx], (res, pose.sequence()[idx])
    return pose


def get_pose(pdb):
    return pose_from_pdb(pdb)


# from PyRosetta toolbox
def restrict_non_nbrs_from_repacking(pose, res, task, pack_radius, repack_rotamers=0):
    """Configure a `PackerTask` to only repack neighboring residues and
    return the task.

    Args:
        pose (pyrosetta.Pose): The `Pose` to opertate on.
        res (int): Pose-numbered residue position to exclude.
        task (pyrosetta.rosetta.core.pack.task.PackerTask): `PackerTask` to modify.
        pack_radius (float): Radius used to define neighboring residues.

    Returns:
        pyrosetta.rosetta.core.pack.task.PackerTask: Configured `PackerTask`.
    """

    if not repack_rotamers:
        assert pack_radius == 0, "pack radius must be 0 if you don't want to repack rotamers"

    def representative_coordinate(resNo):
        return pose.residue(resNo).xyz(pose.residue(resNo).nbr_atom())

    center = representative_coordinate(res)
    for i in range(1, len(pose.residues) + 1):
        # only pack the mutating residue and any within the pack_radius
        if i == res:
            # comment out this block to reproduce biorxiv results 
            #if not repack_rotamers:
            #   task.nonconst_residue_task(i).prevent_repacking()
            continue
        if center.distance(representative_coordinate(i)) > pack_radius:
            task.nonconst_residue_task(i).prevent_repacking()
        else:
            if repack_rotamers:
                task.nonconst_residue_task(i).restrict_to_repacking()
            else:
                task.nonconst_residue_task(i).prevent_repacking()

    return task


# modified from PyRosetta toolbox
def mutate_residue(pose, mutant_position, mutant_aa, pack_radius=0.0, pack_scorefxn=None, repack_rotamers=0):
    """Replace the residue at a single position in a Pose with a new amino acid
        and repack any residues within user-defined radius of selected residue's
        center using.

    Args:
        pose (pyrosetta.rosetta.core.pose.Pose):
        mutant_position (int): Pose-numbered position of the residue to mutate.
        mutant_aa (str): The single letter name for the desired amino acid.
        pack_radius (float): Radius used to define neighboring residues.
        pack_scorefxn (pyrosetta.ScoreFunction): `ScoreFunction` to use when repacking the `Pose`.
            Defaults to the standard `ScoreFunction`.
    """

    wpose = pose  

    if not wpose.is_fullatom():
        raise IOError("mutate_residue only works with fullatom poses")

    # create a standard scorefxn by default
    if not pack_scorefxn:
        pack_scorefxn = pyrosetta.get_score_function()
    
    # forces mutation
    mut = MutateResidue(mutant_position, common.atoms.aa_inv[mutant_aa])
    mut.apply(wpose)
    
    # the numbers 1-20 correspond individually to the 20 proteogenic amino acids
    mutant_aa = int(aa_from_oneletter_code(mutant_aa))
    aa_bool = pyrosetta.Vector1([aa == mutant_aa for aa in range(1, 21)])
    # mutation is performed by using a PackerTask with only the mutant
    # amino acid available during design

    task = pyrosetta.standard_packer_task(wpose)
    task.nonconst_residue_task(mutant_position).restrict_absent_canonical_aas(aa_bool)

    # prevent residues from packing by setting the per-residue "options" of the PackerTask
    task = restrict_non_nbrs_from_repacking(wpose, mutant_position, task, pack_radius, repack_rotamers=repack_rotamers)

    # apply the mutation and pack nearby residues

    packer = PackRotamersMover(pack_scorefxn, task)
    packer.apply(wpose)
    # return pack_or_pose
    return wpose
