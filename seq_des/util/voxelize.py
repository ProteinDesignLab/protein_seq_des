import numpy as np


def voxelize(x, x_data, n=20, c=13, dist=10, plot=False, bb_only=0):
    """Function to voxelize atom coordinate data ahead of training. Could be sped up on GPU
    
    Args:
        x_coords (np.array): num_res x num_return x 1 x 3 -- canonicalized coordinates about each residue with center residue masked
        x_data (np.array): num_res x num_return x 1 x 4 -- metadata for canonicalized atoms for each environment 
    Returns:
        output (np.array): num_res x c x n x n x n -- 3D environments centered at each residue with atom type in channel dimensions 
    """

    bins = np.linspace(-dist, dist, n + 1)
    bs, nres, _, x_dim = x.shape
    x_data_dim = x_data.shape[-1]
    x = x.reshape(bs * nres, -1, x_dim)
    x_data = x_data.reshape(bs * nres, -1, x_data_dim)
    x_atom = x_data[..., 2].astype(np.int64)
    x_res_type = x_data[..., -1].astype(np.int64)
    x_bb = x_data[..., 1].astype(np.int64)

    bs_idx = np.tile(np.arange(bs)[:, None], (1, nres)).reshape(-1)
    # coordinates to voxels
    x_b = np.digitize(x[..., 0], bins)  # [:, 0]
    y_b = np.digitize(x[..., 1], bins)  # [:, 0]
    z_b = np.digitize(x[..., 2], bins)  # [:, 0]

    # eliminate 'other' atoms
    x_atom[x_atom > c - 1] = c  # force any non-listed atoms into 'other' category

    # this step can possibly be moved to GPU
    output_atom = np.zeros((bs, c + 1, n + 2, n + 2, n + 2))
    output_atom[bs_idx, x_atom[:, 0], x_b[:, 0], y_b[:, 0], z_b[:, 0]] = 1  # atom type
    if not bb_only:
        output_bb = np.zeros((bs, 2, n + 2, n + 2, n + 2))
        output_bb[bs_idx, x_bb[:, 0], x_b[:, 0], y_b[:, 0], z_b[:, 0]] = 1  # BB indicator
        output_res = np.zeros((bs, 22, n + 2, n + 2, n + 2))
        output_res[bs_idx, x_res_type[:, 0], x_b[:, 0], y_b[:, 0], z_b[:, 0]] = 1  # res type for each atom
        # eliminate last channel for output_atom ('other' atom type), output_bb, and output_res (res type for current side chain)
        output = np.concatenate([output_atom[:, :c], output_bb[:, :1], output_res[:, :21]], 1)
    else:
        output = output_atom[:, :c]

    output = output[:, :, 1:-1, 1:-1, 1:-1]

    return output


def get_voxel_idx(x, x_data, n=20, c=13, dist=10, plot=False):
    """Function to get indices for voxelized atom coordinate data ahead of training.
    
    Args:
        x_coords (np.array): num_res x num_return x 1 x 3 -- canonicalized coordinates about each residue with center residue masked
        x_data (np.array): num_res x num_return x 1 x 4 -- metadata for canonicalized atoms for each environment 
    Returns:
        #NOTE -- FIX THIS
        output (np.array): num_res x c x n x n x n -- 3D environments centered at each residue with atom type in channel dimensions 
    """

    bins = np.linspace(-dist, dist, n + 1)
    bs, nres, _, x_dim = x.shape
    x_data_dim = x_data.shape[-1]
    x = x.reshape(bs * nres, -1, x_dim)
    x_data = x_data.reshape(bs * nres, -1, x_data_dim)
    x_atom = x_data[..., 2].astype(np.int64)
    x_res_type = x_data[..., -1].astype(np.int64)  # not used for now
    x_bb = x_data[..., 1].astype(np.int64)

    bs_idx = np.tile(np.arange(bs)[:, None], (1, nres)).reshape(-1)

    # coordinates to voxels
    x_b = np.digitize(x[..., 0], bins)  # [:, 0]
    y_b = np.digitize(x[..., 1], bins)  # [:, 0]
    z_b = np.digitize(x[..., 2], bins)  # [:, 0]

    # eliminate 'other' atoms
    x_atom[x_atom > c - 1] = c  # force any non-listed atoms into 'other' category
    # print(x_atom.shape, x_res_type.shape, x_bb.shape)

    return bs_idx, x_atom[..., 0], x_bb[..., 0], x_b[..., 0], y_b[..., 0], z_b[..., 0], x_res_type[..., 0]


