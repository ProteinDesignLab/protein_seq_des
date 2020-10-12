import numpy as np
import copy
import glob
import pickle


gly_CB_mu = np.array([-0.5311191 , -0.75842446,  1.2198311 ]) #pickle.load(open("pkl/CB_mu.pkl", "rb"))


def get_len(v):
    return np.sqrt(np.sum(v ** 2, -1))


def get_unit_normal(ab, bc):
    n = np.cross(ab, bc, -1)
    length = get_len(n)
    if len(n.shape) > 2:
        length = length[..., None]
    return n / length


def get_angle(v1, v2):
    # get in plane angle between v1, v2 -- cos^-1(v1.v2 / ||v1|| ||v2||)
    return np.arccos(np.sum(v1 * v2, -1) / get_len(v1) * get_len(v2))


def bdot(a, b):
    return np.matmul(a, b)


def return_align_f(axis, theta):
    c_theta = np.cos(theta)[..., None]
    s_theta = np.sin(theta)[..., None]
    f_rot = lambda v: c_theta * v + s_theta * np.cross(axis, v, axis=-1) + (1 - c_theta) * bdot(axis, v.transpose(0, 2, 1)) * axis
    return f_rot


def return_batch_align_f(axis, theta, n):
    # n is total number of atoms
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    axis = np.repeat(axis, n, axis=1)[:, :, None]
    c_theta = np.repeat(c_theta, n, axis=1)[:, :, None, None]
    s_theta = np.repeat(s_theta, n, axis=1)[:, :, None, None]

    f_rot = lambda v: c_theta * v + s_theta * np.cross(axis, v, axis=-1) + (1 - c_theta) * bdot(axis, v.transpose(0, 1, 3, 2)) * axis
    return f_rot


def get_batch_N_CA_C_align(normal, r, n):
    # get fn to align n to positive z_hat, via rotation about x axis (assume N-CA already along x_hat)
    # r is number of residues
    z = np.repeat(np.array([[0, 0, 1]]), r, 0)[:, None]
    theta = get_angle(normal, z)
    axis = get_unit_normal(normal, z)
    return return_align_f(axis, theta), return_batch_align_f(axis, theta, n=n)


def get_batch_N_CA_align(v, r, n):
    # assuming ca is at (0,0,0), return fn to batch align CA--N to positive x axis
    # v = n - ca
    x = np.repeat(np.array([[1, 0, 0]])[None], r, 0)
    axis = get_unit_normal(v, x)
    theta = get_angle(v, x)
    return return_align_f(axis, theta), return_batch_align_f(axis, theta, n=n)


def batch_canonicalize_coords(atom_coords, atom_data, residue_bb_index_list, res_idx=None, num_return=400, bb_only=0):
    """Function to get batch canonicalize atoms about all residues in a structure and mask out residue of interest.
    
    Args:
        atom_coords (np.array): num_atoms x 3 coordinates of all retained atoms in structure 
        atom_data (np.array): num_atoms x 4 data for atoms  -- [residue idx, BB ind, atom type, res type] 
        residue_bb_index_list (np.array): num_res x 4 mapping from residue idx to atom indices for backbone atoms (N, CA, C, CB) used for canonicalization
        res_idx (np.array): num_output_res x 1 -- residue indices for subsampling residues ahead of canonicalization   
        num_return (int): number of atoms to preserve about residue in environment
    Returns:
        x_coords (np.array): num_output_res x num_return x 1 x 3 -- canonicalized coordinates about each residue with center residue masked
        x_data (np.array): num_output_res x num_return x 1 x 4 -- metadata for canonicalized atoms for each environment
    """

    n_atoms = atom_coords.shape[0]

    # subsampling residues to canonicalize
    if res_idx is not None:
        residue_bb_index_list = residue_bb_index_list[res_idx]
        n_res = len(res_idx)
    else:
        n_res = residue_bb_index_list.shape[0]

    num_return = min(num_return, n_atoms - 15)

    idx_N, idx_CA, idx_C, idx_CB = residue_bb_index_list[:, 0], residue_bb_index_list[:, 1], residue_bb_index_list[:, 2], residue_bb_index_list[:, 3]
    x = atom_coords.copy()

    center = x[idx_CA].copy()
    x_idxN, x_idxC, x_idxCA, x_idxCB = x[idx_N] - center, x[idx_C] - center, x[idx_CA] - center, x[idx_CB] - center
    x_data = atom_data.copy()

    x = np.repeat(x[None], n_res, axis=0)
    x_data = np.repeat(x_data[None], n_res, axis=0)

    # center coordinates at CA position
    x = x - center[:, None]

    # for each residue, eliminate side chain residue coordinates here --
    bs, _, _, x_dim = x.shape
    x_data_dim = x_data.shape[-1]

    if res_idx is None:
        res_idx = np.arange(n_res)

    res_idx = np.tile(res_idx[:, None], (1, n_atoms)).reshape(-1)
    x = x.reshape(-1, x_dim)
    x_data = x_data.reshape(-1, x_data_dim)
    # get res_idx, indicator of bb atom
    x_res, x_bb, x_res_type = x_data[..., 0], x_data[..., 1], x_data[..., -1]
    assert len(x_res) == len(res_idx)

    if not bb_only:
        # exclude atoms on residue of interest that are not BB atoms
        exclude_idx = np.where((x_res == res_idx) & (x_bb != 1))[0]
    else:
        # exclude all side-chain atoms (bb only)
        exclude_idx = np.where((x_bb != 1))[0]

    # mask res type for all current residue atoms (no cheating!)
    res_type_exclude_idx = np.where((x_res == res_idx))[0]
    x_res_type[res_type_exclude_idx] = 21  # set to idx higher than highest --

    # move coordinates for non-include residues well out of frame of reference -- will be omitted in next step or voxelize
    x[exclude_idx] = x[exclude_idx] + np.array([-1000.0, -1000.0, -1000.0])
    x = x.reshape(bs, n_atoms, x_dim)[:, :, None]

    x_data = x_data.reshape(bs, n_atoms, x_data_dim)[:, :, None]

    # select num_return nearest atoms to env center
    d_x_out = np.sqrt(np.sum(x ** 2, -1))
    idx = np.argpartition(d_x_out, kth=num_return, axis=1)
    idx = idx[:, :num_return]

    x = np.take_along_axis(x, idx[..., None], axis=1)
    x_data = np.take_along_axis(x_data, idx[..., None], axis=1)

    n = num_return

    # align N-CA along x axis
    f_R, f_bR = get_batch_N_CA_align(x_idxN - x_idxCA, r=n_res, n=n)  # um_return)
    x = f_bR(x)
    x_idxN, x_idxC, x_idxCA, x_idxCB = f_R(x_idxN), f_R(x_idxC), f_R(x_idxCA), f_R(x_idxCB)

    # rotate so that normal of N-CA-C plane aligns to positive z_hat
    normal = get_unit_normal(x_idxN, x_idxC)
    f_R, f_bR = get_batch_N_CA_C_align(normal, r=n_res, n=n)  # um_return)
    x_idxN, x_idxC, x_idxCA, x_idxCB = f_R(x_idxN), f_R(x_idxC), f_R(x_idxCA), f_R(x_idxCB)
    x = f_bR(x)

    # recenter at CB
    fixed_CB = np.ones((x_idxCB.shape[0], 1, 3)) * gly_CB_mu
    x = x - fixed_CB[:, None]

    return x, x_data
