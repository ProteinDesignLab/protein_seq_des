import Bio.PDB
import Bio.PDB.vectors

import torch
from torch.utils import data
import torch.nn.functional as F

import json
import numpy as np
import os
import re
import glob

import common.atoms
import seq_des.util.canonicalize as canonicalize
import seq_des.util.voxelize as voxelize


CHI_BINS = np.linspace(-np.pi, np.pi, num=25)

def read_domain_ids_per_chain_from_txt(txt_file):
    pdbs = []
    ids_chains = {}
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip('\n').split()
            pdbs.append(line[0][:4])
            ids_chains[line[0][:4]] = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip('\n').split()
            if len(line) == 6: # no icodes
                line.extend([' ', ' '])
            elif len(line) == 7:
                line.extend([' '])
            pdb = line[0][:4]
            ids_chains[pdb].append(tuple(line)) #line[:4], line[4:]))
    return [(k, ids_chains[k]) for k in ids_chains.keys()]


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



def get_domain_envs(pdb_id, domains_list, pdb_dir="/data/drive2tb/protein/pdb", num_return=400, bb_only=0):
    """ Get domain specific residues and local environments by first getting full biological assembly for 
        pdb of interest -- selecting domain specific residues.
    
    Args:
        pdb_id (str): pdb structure ID
        domains_list (list of list of tuples of str): for each domain within pdb of interest -- list of domain start, stop residue IDs and icodes

    Returns:
        atom_coords_canonicalized (np.array): n_res x n_atoms x 3 array with canonicalized local 
                                              environment atom coordinates 
        atom_data_canonicalized (np.array): n_res x n_atoms x 4 with metadata for local env atoms
                                            [residue idx, BB ind, atom type, res type]
        res_data (dict of list of lists): dictionary with residue metadata -- {chain ID: [ [residue ID, residue icode, residue index, residue type], ...]}
        res_label (np.array): num_res x 1 residue type labels (amino acid type) for all residues (to be included in training)

    """

    atom_coords, atom_data, residue_bb_index_list, res_data, res_label, chis = get_pdb_data(pdb_id, data_dir=pdb_dir)
    atom_coords_def = None

    assert len(res_label) > 0

    ind_assembly = []
    res_idx_list_domains = []
    # iterate over domains for PDB of interest
    for domain_split in domains_list:
        domain_id = domain_split[0]
        domain_split = domain_split[-1]
        chain_id, domains = get_domain(domain_split)
        res_idx_list = []
        # iterate over start/end cutpoints for domain
        if chain_id in res_data.keys():
            ind_assembly.append(1)
        else:
            if atom_coords_def is None:
                atom_coords_def, atom_data_def, residue_bb_index_list_def, res_data_def, res_label_def, chis_def = get_pdb_data(pdb_id, data_dir=pdb_dir, assembly=0)

            if chain_id in res_data_def.keys():
                ind_assembly.append(0)
                if atom_coords_def is None:
                    atom_coords_def, atom_data_def, residue_bb_index_list_def, res_data_def, res_label_def, chis_def = get_pdb_data(pdb_id, data_dir=pdb_dir, assembly=0)
            else:
                print("chain not found", chain_id, res_data.keys(), res_data_def.keys())
                continue
        for ds, de in domains:
            start = False
            end = False
            if chain_id not in res_data.keys():
                for res_id, res_icode, res_idx, res_type in res_data_def[chain_id]:
                    assert res_idx < len(res_label_def)
                    if (res_id != ds) and not start:
                        continue
                    elif res_id == ds:
                        start = True
                    if res_id == de:
                        end = True
                    if start and not end:
                        res_idx_list.append(res_idx)
                    if end:
                        break
            else:
                # parse chain_res_data to get res_idx for domain of interest
                for res_id, res_icode, res_idx, res_type in res_data[chain_id]:
                    assert res_idx < len(res_label)
                    if (res_id != ds) and not start:
                        continue
                    elif res_id == ds:
                        start = True
                    if res_id == de:
                        end = True
                    if start and not end:
                        res_idx_list.append(res_idx)
                    if end:
                        break
        res_idx_list_domains.append(res_idx_list)

    assert len(res_idx_list_domains) == len(ind_assembly)

    atom_coords_out = []
    atom_data_out = []
    res_label_out = []
    domain_ids_out = []
    chis_out = []
    
    for i in range(len(res_idx_list_domains)):
        # canonicalize -- subset of residues
        if len(res_idx_list_domains[i]) == 0:
            continue
        if ind_assembly[i] == 1:
            # pull data from biological assembly
            atom_coords_canonicalized, atom_data_canonicalized = canonicalize.batch_canonicalize_coords(atom_coords, atom_data, residue_bb_index_list, res_idx=np.array(res_idx_list_domains[i]), num_return=num_return, bb_only=bb_only)
        else:
            # pull data from default structure
            atom_coords_canonicalized, atom_data_canonicalized = canonicalize.batch_canonicalize_coords(
                atom_coords_def, atom_data_def, residue_bb_index_list_def, res_idx=np.array(res_idx_list_domains[i]), num_return=num_return, bb_only=bb_only
            )

        atom_coords_out.append(atom_coords_canonicalized)
        atom_data_out.append(atom_data_canonicalized)
        if ind_assembly[i] == 1:
            res_label_out.append(res_label[res_idx_list_domains[i]])
            assert len(atom_coords_canonicalized) == len(res_label[res_idx_list_domains[i]])
            chis_out.append(chis[res_idx_list_domains[i]])
        else:
            res_label_out.append(res_label_def[res_idx_list_domains[i]])
            assert len(atom_coords_canonicalized) == len(res_label_def[res_idx_list_domains[i]])
            chis_out.append(chis_def[res_idx_list_domains[i]])
        domain_ids_out.append(domains_list[i][0])

    return atom_coords_out, atom_data_out, res_label_out, domain_ids_out, chis_out


def get_domain(domain_split):
    # function to parse CATH domain info from txt -- returns chain and domain residue IDs
    chain = domain_split[-1]

    domains = domain_split.split(",")
    domains = [d[: d.rfind(":")] for d in domains]

    domains = [(d[: d.rfind("-")], d[d.rfind("-") + 1 :]) for d in domains]
    domains = [(int(re.findall("\D*\d+", ds)[0]), int(re.findall("\D*\d+", de)[0])) for ds, de in domains]

    return chain, np.array(domains)



class PDB_domain_spitter(data.Dataset):
    def __init__(self, txt_file="data/052320_cath-b-newest-all.txt", pdb_path="/data/drive2tb/protein/pdb", num_return=400, bb_only=0):
        self.domains = read_domain_ids_per_chain_from_txt(txt_file)
        self.pdb_path = pdb_path
        self.num_return = num_return
        self.bb_only = bb_only

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, index):
        pdb_id, domain_list = self.domains[index]
        return self.get_data(pdb_id, domain_list)

    def get_and_download_pdb(self, index):
        pdb_id, domain_list = self.domains[index]
        f = download_pdb(pdb_id, data_dir=self.pdb_path)
        return f

    def get_data(self, pdb, domain_list):
        try:
            atom_coords, atom_data, res_label, domain_id, chis = get_domain_envs(pdb, domain_list, pdb_dir=self.pdb_path, num_return=self.num_return, bb_only=self.bb_only)
            return atom_coords, atom_data, res_label, domain_id, chis
        except:
            return [] 


class PDB_data_spitter(data.Dataset):
    def __init__(self, data_dir="/data/simdev_2tb/protein/sequence_design/data/coords/test_s95_chi/", n=20, dist=10, datalen=1000):
        self.files = glob.glob("%s/data*pt" % (data_dir))
        self.cached_pt = -1
        self.chunk_size = 10000  # args.chunk_size #50000i #NOTE -- CAUTION
        self.datalen = datalen
        self.data_dir = data_dir
        self.n = n
        self.dist = dist
        self.c = len(common.atoms.atoms)
        self.len = 0

    def __len__(self):
        if self.len == 0:
            return len(self.files) * self.chunk_size
        else:
            return self.len

    def get_data(self, index):
        if self.cached_pt != index // self.chunk_size:
            self.cached_pt = int(index // self.chunk_size)
            self.xs, self.x_data, self.ys, self.domain_ids, self.chis = torch.load("%s/data_%0.4d.pt" % (self.data_dir, self.cached_pt))

        index = index % self.chunk_size
        x, x_data, y, domain_id, chis = self.xs[index], self.x_data[index], self.ys[index], self.domain_ids[index], self.chis[index]
        return x, x_data, y, domain_id, chis

    def __getitem__(self, index):  # index):
        x, x_data, y, domain_id, chis = self.get_data(index)
        ## voxelize coordinates and atom metadata
        bs_idx, x_atom, x_bb, x_b, y_b, z_b, x_res_type = voxelize.get_voxel_idx(x[None], x_data[None], n=self.n, c=self.c, dist=self.dist)
        # map chi angles to bins
        chi_angles = chis[0]
        chi_mask = chis[1]
        chi_angles_binned = map_to_bins(chi_angles)
        chi_angles_binned[chi_mask == 0] = 0 # ignore index

        # return domain_id, x, x_data, y, chi_angles, chi_angles_binned
        return bs_idx, x_atom, x_bb, x_b, y_b, z_b, x_res_type, y, chi_angles, chi_angles_binned


def collate_wrapper(data, crop=True):
    max_n = 0
    for i in range(len(data)):
        bs_idx, x_atom, x_bb, x_b, y_b, z_b, x_res_type, y, chi_angles, chi_angles_binned = data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7], data[i][8], data[i][9]
        # print(bs_idx.shape, x_atom.shape, x_bb.shape, x_b.shape, y_b.shape, z_b.shape, x_res_type.shape)# if pwd is greater than CROP_SIZE -- random crop
        n_i = x_atom.shape[-1]
        # print(n_i, min_n)
        if n_i > max_n:
            max_n = n_i

    # pad pwd data, coords
    out_bs_idx = []
    out_y = []
    out_x_atom = []
    out_x_bb = []
    out_x_b = []
    out_y_b = []
    out_z_b = []
    out_x_res_type = []
    out_chi_angles = []
    out_chi_angles_binned = []
    padding = False
    for i in range(len(data)):
        bs_idx, x_atom, x_bb, x_b, y_b, z_b, x_res_type, y, chi_angles, chi_angles_binned = data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7], data[i][8], data[i][9]
        n_i = x_atom.shape[-1]

        if n_i < max_n:
            padding = True
            # zero pad all --> x, y, z indexing will be omitted
            x_atom = np.pad(x_atom, ((0, max_n - n_i)), mode='constant')
            x_b = np.pad(x_b, ((0, max_n - n_i)), mode='constant')
            y_b = np.pad(y_b, ((0, max_n - n_i)), mode='constant')
            z_b = np.pad(z_b, ((0, max_n - n_i)), mode='constant')
            x_bb = np.pad(x_bb, ((0, max_n - n_i)), mode='constant')
            x_res_type = np.pad(x_res_type, ((0, max_n - n_i)), mode='constant')

        # handle batch indexing correctly
        out_bs_idx.append(torch.Tensor([i for j in range(len(x_b))])[None])
        out_y.append(torch.Tensor([y]))  # [None])
        out_x_atom.append(torch.Tensor(x_atom)[None])
        out_x_bb.append(torch.Tensor(x_bb)[None])
        out_x_b.append(torch.Tensor(x_b)[None])
        out_y_b.append(torch.Tensor(y_b)[None])
        out_z_b.append(torch.Tensor(z_b)[None])
        out_x_res_type.append(torch.Tensor(x_res_type)[None])
        out_chi_angles.append(torch.Tensor(chi_angles)[None])
        out_chi_angles_binned.append(torch.Tensor(chi_angles_binned)[None])

    out_bs_idx = torch.cat(out_bs_idx, 0)
    out_y = torch.cat(out_y, 0)
    out_x_atom = torch.cat(out_x_atom, 0)
    out_x_bb = torch.cat(out_x_bb, 0)
    out_x_b = torch.cat(out_x_b, 0)
    out_y_b = torch.cat(out_y_b, 0)
    out_z_b = torch.cat(out_z_b, 0)
    out_x_res_type = torch.cat(out_x_res_type, 0)
    out_chi_angles = torch.cat(out_chi_angles, 0)
    out_chi_angles_binned = torch.cat(out_chi_angles_binned, 0)
    return out_bs_idx.long(), out_x_atom.long(), out_x_bb.long(), out_x_b.long(), out_y_b.long(), out_z_b.long(), out_x_res_type.long(), out_y.long(), out_chi_angles, out_chi_angles_binned.long()


