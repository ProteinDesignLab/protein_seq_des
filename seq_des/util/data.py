import Bio.PDB
import Bio.PDB.vectors

import torch
from torch.utils import data
import torch.nn.functional as F

import json
import numpy as np
import os

import common.atoms
import seq_des.util.canonicalize as canonicalize
import seq_des.util.voxelize as voxelize


CHI_BINS = np.linspace(-np.pi, np.pi, num=25)


def map_to_bins(chi):
    # map rotamer angles to discretized bins
    binned_pwd = np.digitize(chi, CHI_BINS)
    if len(binned_pwd[binned_pwd == 0]) > 0:
        binned_pwd[binned_pwd == 0] = 1  # in case chi == -np.pi
    return binned_pwd


def download_pdb(pdb, data_dir, assembly=1):

    """Function to download pdb -- either biological assembly or if that
    is not available/specified -- download default pdb structure
    Uses biological assembly as default, otherwise gets default pdb.
    
    Args:
        pdb (str): pdb ID.
        data_dir (str): path to pdb directory

    Returns:
        f (str): path to downloaded pdb

    """

    if assembly:
        f = data_dir + "/" + pdb + ".pdb1"
        if not os.path.isfile(f):
            try:
                os.system("wget -O {}.gz https://files.rcsb.org/download/{}.pdb1.gz".format(f, pdb.upper()))
                os.system("gunzip {}.gz".format(f))

            except:
                f = data_dir + "/" + pdb + ".pdb"
                if not os.path.isfile(f):
                    os.system("wget -O {} https://files.rcsb.org/download/{}.pdb".format(f, pdb.upper()))
    else:
        f = data_dir + "/" + pdb + ".pdb"

    if not os.path.isfile(f):
        os.system("wget -O {} https://files.rcsb.org/download/{}.pdb".format(f, pdb.upper()))

    return f


def get_pdb_chains(pdb, data_dir, assembly=1, skip_download=0):

    """Function to load pdb structure via Biopython and extract all chains. 
    Uses biological assembly as default, otherwise gets default pdb.
    
    Args:
        pdb (str): pdb ID.
        data_dir (str): path to pdb directory

    Returns:
        chains (list of (chain, chain_id)): all pdb chains

    """
    if not skip_download:
        f = download_pdb(pdb, data_dir, assembly=assembly)

    if assembly:
        f = data_dir + "/" + pdb + ".pdb1"
        if not os.path.isfile(f):
            f = data_dir + "/" + pdb + ".pdb"
    else:
        f = data_dir + "/" + pdb + ".pdb"

    assert os.path.isfile(f)
    structure = Bio.PDB.PDBParser(QUIET=True).get_structure(pdb, f)

    assert len(structure) > 0, pdb

    # for assemblies -- sometimes chains are represented as different structures
    if len(structure) > 1:
        model = structure[0]
        count = 0
        for i in range(len(structure)):
            for c in structure[i].get_chains():
                try:
                    c.id = common.atoms.rename_chains[count]
                except:
                    continue
                count += 1
                try:
                    model.add(c)
                except Bio.PDB.PDBExceptions.PDBConstructionException:
                    continue
    else:
        model = structure[0]

    # special hard-coded case with very large assembly -- not necessary to train on all
    if "2y26" in pdb:
        return [(c, c.id) for c in model.get_chains() if c.id in ["B", "A", "E", "C", "D"]]

    return [(c, c.id) for c in model.get_chains()]


def get_pdb_data(pdb, data_dir="", assembly=1, skip_download=0):

    """Function to get atom coordinates and atom/residue metadata from pdb structures. 
    
    Args:
        pdb (str): pdb ID
        data_dir (str): path to pdb directory
        assembly (int): 0/1 indicator of whether to use biological assembly or default pdb
        skip_download (int): 0/1 indicator of whether to skip attempt to download pdb from remote server

    Returns:
        atom_coords (np.array): num_atoms x 3 coordinates of all retained atoms in structure
        atom_data (np.array): num_atoms x 4 data for atoms  -- [residue idx, BB ind, atom type, res type]
        residue_bb_index_list (np.array): num_res x 4 mapping from residue idx to atom indices for backbone atoms (N, CA, C, CB) used for canonicalization 
        res_data (dict of list of lists): dictionary {chain ID: [ [residue ID, residue icode, residue index, residue type], ...]}
        res_label (np.array): num_res x 1 residue type labels (amino acid type) for all residues (to be included in training)

    """

    # get pdb chain data
    pdb_chains = get_pdb_chains(pdb, data_dir, assembly=assembly, skip_download=skip_download)

    res_idx = 0
    res_data = {}
    atom_coords = []
    atom_data = []
    residue_bb_index = {}
    residue_bb_index_list = []
    res_label = []
    chis = []
    # iterate over chains
    for pdb_chain, chain_id in pdb_chains:
        # iterate over residues
        res_data[chain_id] = []
        for res in pdb_chain.get_residues():
            skip_res = False  # whether to skip training directly on this residue

            res_name = res.get_resname()
            het, res_id, res_icode = res.id

            # skip waters, metal ions, pre-specified ligands, unknown ligands
            if res_name in common.atoms.skip_res_list:
                continue

            res_atoms = [atom for atom in res.get_atoms()]

            # skip training on residues where all BB atoms are not present -- this will break canonicalization
            if res_name in common.atoms.res_label_dict.keys() and len(res_atoms) < 4:
                skip_res = True

            # if residue is an amino acid, add to label and save residue ID
            if (not skip_res) and (res_name in common.atoms.res_label_dict.keys()):
                res_type = common.atoms.res_label_dict[res_name]
                res_data[chain_id].append((res_id, res_icode, res_idx, res_type))
                res_label.append(res_type)
            residue_bb_index[res_idx] = {}

            # iterate over atoms -- get coordinate data
            for atom in res.get_atoms():

                if atom.element in common.atoms.skip_atoms:
                    continue
                elif atom.element not in common.atoms.atoms:
                    if res_name == "MSE" and atom.element == "SE":
                        elem_name = "S"  # swap MET for MSE
                    else:
                        elem_name = "other"  # all other atoms are labeled 'other'
                else:
                    elem_name = atom.element

                # get atomic coordinate
                c = np.array(list(atom.get_coord()))[None].astype(np.float32)

                # get atom type index
                assert elem_name in common.atoms.atoms
                atom_type = common.atoms.atoms.index(elem_name)

                # get whether atom is a BB atom
                bb = int(res_name in common.atoms.res_label_dict.keys() and atom.name in ["N", "CA", "C", "O", "OXT"])

                if res_name in common.atoms.res_label_dict.keys():
                    res_type_idx = common.atoms.res_label_dict[res_name]
                else:
                    res_type_idx = 20  # 'other' type (ligand, ion)

                # index -- residue idx, bb?, atom index, residue type (AA)
                index = np.array([res_idx, bb, atom_type, res_type_idx])
                atom_coords.append(c)
                atom_data.append(index[None])
                # if atom is BB atom, add to residue_bb_index dictionary
                if (not skip_res) and ((res_name in common.atoms.res_label_dict.keys())):
                    # map from residue index to atom coordinate
                    residue_bb_index[res_idx][atom.name] = len(atom_coords) - 1

            # get rotamer chi angles
            if (not skip_res) and (res_name in common.atoms.res_label_dict.keys()):
                if res_name == "GLY" or res_name == "ALA":
                    chi = [0, 0, 0, 0]
                    mask = [0, 0, 0, 0]

                else:
                    chi = []
                    mask = []
                    if "N" in residue_bb_index[res_idx].keys() and "CA" in residue_bb_index[res_idx].keys():
                        n = Bio.PDB.vectors.Vector(list(atom_coords[residue_bb_index[res_idx]["N"]][0]))
                        ca = Bio.PDB.vectors.Vector(list(atom_coords[residue_bb_index[res_idx]["CA"]][0]))
                        if (
                            "chi_1" in common.atoms.chi_dict[common.atoms.label_res_dict[res_type]].keys()
                            and common.atoms.chi_dict[common.atoms.label_res_dict[res_type]]["chi_1"] in residue_bb_index[res_idx].keys()
                            and "CB" in residue_bb_index[res_idx].keys()
                        ):
                            cb = Bio.PDB.vectors.Vector(list(atom_coords[residue_bb_index[res_idx]["CB"]][0]))
                            cg = Bio.PDB.vectors.Vector(
                                atom_coords[residue_bb_index[res_idx][common.atoms.chi_dict[common.atoms.label_res_dict[res_type]]["chi_1"]]][0]
                            )
                            chi_1 = Bio.PDB.vectors.calc_dihedral(n, ca, cb, cg)
                            chi.append(chi_1)
                            mask.append(1)

                            if (
                                "chi_2" in common.atoms.chi_dict[common.atoms.label_res_dict[res_type]].keys()
                                and common.atoms.chi_dict[common.atoms.label_res_dict[res_type]]["chi_2"] in residue_bb_index[res_idx].keys()
                            ):
                                cd = Bio.PDB.vectors.Vector(
                                    atom_coords[residue_bb_index[res_idx][common.atoms.chi_dict[common.atoms.label_res_dict[res_type]]["chi_2"]]][0]
                                )
                                chi_2 = Bio.PDB.vectors.calc_dihedral(ca, cb, cg, cd)
                                chi.append(chi_2)
                                mask.append(1)

                                if (
                                    "chi_3" in common.atoms.chi_dict[common.atoms.label_res_dict[res_type]].keys()
                                    and common.atoms.chi_dict[common.atoms.label_res_dict[res_type]]["chi_3"] in residue_bb_index[res_idx].keys()
                                ):
                                    ce = Bio.PDB.vectors.Vector(
                                        atom_coords[residue_bb_index[res_idx][common.atoms.chi_dict[common.atoms.label_res_dict[res_type]]["chi_3"]]][
                                            0
                                        ]
                                    )
                                    chi_3 = Bio.PDB.vectors.calc_dihedral(cb, cg, cd, ce)
                                    chi.append(chi_3)
                                    mask.append(1)

                                    if (
                                        "chi_4" in common.atoms.chi_dict[common.atoms.label_res_dict[res_type]].keys()
                                        and common.atoms.chi_dict[common.atoms.label_res_dict[res_type]]["chi_4"] in residue_bb_index[res_idx].keys()
                                    ):
                                        cz = Bio.PDB.vectors.Vector(
                                            atom_coords[
                                                residue_bb_index[res_idx][common.atoms.chi_dict[common.atoms.label_res_dict[res_type]]["chi_4"]]
                                            ][0]
                                        )
                                        chi_4 = Bio.PDB.vectors.calc_dihedral(cg, cd, ce, cz)
                                        chi.append(chi_4)
                                        mask.append(1)
                                    else:
                                        chi.append(0)
                                        mask.append(0)
                                else:
                                    chi.extend([0, 0])
                                    mask.extend([0, 0])

                            else:
                                chi.extend([0, 0, 0])
                                mask.extend([0, 0, 0])
                        else:
                            chi = [0, 0, 0, 0]
                            mask = [0, 0, 0, 0]
                    else:
                        chi = [0, 0, 0, 0]
                        mask = [0, 0, 0, 0]
                chi = np.array(chi)
                mask = np.array(mask)
                chis.append(np.concatenate([chi[None], mask[None]], axis=0))

            # add bb atom indices in residue_list to residue_bb_index dict
            if (not skip_res) and res_name in common.atoms.res_label_dict.keys():
                residue_bb_index[res_idx]["list"] = []
                for atom in ["N", "CA", "C", "CB"]:
                    if atom in residue_bb_index[res_idx]:
                        residue_bb_index[res_idx]["list"].append(residue_bb_index[res_idx][atom])
                    else:
                        # GLY handling for CB
                        residue_bb_index[res_idx]["list"].append(-1)

                residue_bb_index_list.append(residue_bb_index[res_idx]["list"])
            if not skip_res and (res_name in common.atoms.res_label_dict.keys()):
                res_idx += 1

    assert len(atom_coords) == len(atom_data)
    assert len(residue_bb_index_list) == len(res_label)
    assert len(chis) == len(residue_bb_index_list)

    return np.array(atom_coords), np.array(atom_data), np.array(residue_bb_index_list), res_data, np.array(res_label), np.array(chis)
