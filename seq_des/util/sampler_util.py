import torch
import numpy as np

import seq_des.util.data as data
import seq_des.util.canonicalize as canonicalize
import seq_des.util.voxelize as voxelize
import common.atoms

import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def get_idx(filename):
    # get variable or fixed indices from list
    with open(filename, "r") as f:
        lines = list(f)
    idx = [int(line.strip("\n").split()[0]) for line in lines]
    return idx


def get_CB_distance(x, x_data):
    # get CB-CB pairwise distances
    A = []
    for k in range(x_data.shape[0]):
        idx_CA, idx_CB = x_data[k, 1], x_data[k, -1]
        if idx_CB >= 0:
            A.append(x[idx_CB])
        else:
            A.append(x[idx_CA])
    A = np.array(A)[:, 0, :3]
    D = np.sqrt(np.sum((A[:, None].repeat(len(A), axis=1) - A[None].repeat(len(A), axis=0)) ** 2, -1))
    return D


def get_graph_from_D(D, threshold):
    A = np.zeros_like(D)
    A[D < threshold] = 1
    return A


def make_onehot(bs, dim, scatter_tensor, use_cuda=1):
    onehot = torch.FloatTensor(bs, dim)
    onehot.zero_()
    onehot.scatter_(1, scatter_tensor, 1)
    if use_cuda:
        return onehot.cuda()
    else:
        return onehot


def get_energy_from_logits(logits, res_idx, mask=None, baseline=0):
    # get negative log prob from logits
    log_p = -F.log_softmax(logits, -1).gather(1, res_idx[:, None])
    if mask is not None:
        log_p[mask == 1] = baseline
    log_p_mean = log_p.mean()
    return log_p, log_p_mean


def get_conv_feat(
    curr_models, atom_coords, atom_data, residue_bb_index_list, res_data, res_label, chis, bb_only=0, return_chi=0, use_cuda=1
):  
    atom_coords_canonicalized, atom_data_canonicalized = canonicalize.batch_canonicalize_coords(
        atom_coords, atom_data, residue_bb_index_list, bb_only=bb_only
    )

    x = atom_coords_canonicalized
    y = res_label
    x_data = atom_data_canonicalized

    voxels = voxelize.voxelize(x, x_data, n=20, c=len(common.atoms.atoms), dist=10, bb_only=bb_only)
    voxels = torch.FloatTensor(voxels)
    bs_i = voxels.size()[0]
    if use_cuda:
        voxels = voxels.cuda()

    # map chi angles to bins
    chi_angles = chis[:, 0]
    chi_mask = chis[:, 1]
    chi_angles_binned = data.map_to_bins(chi_angles)
    chi_angles_binned[chi_mask == 0] = 0
    chi_angles_binned = torch.LongTensor(chi_angles_binned)

    chi_1 = chi_angles_binned[..., 0]
    chi_2 = chi_angles_binned[..., 1]
    chi_3 = chi_angles_binned[..., 2]
    chi_4 = chi_angles_binned[..., 3]

    # get chi onehot vectors -- NOTE can make this faster by precomputing, saving zero tensors
    chi_1_onehot = make_onehot(bs_i, len(data.CHI_BINS), chi_1[:, None], use_cuda=use_cuda)
    chi_2_onehot = make_onehot(bs_i, len(data.CHI_BINS), chi_2[:, None], use_cuda=use_cuda)
    chi_3_onehot = make_onehot(bs_i, len(data.CHI_BINS), chi_3[:, None], use_cuda=use_cuda)
    chi_4_onehot = make_onehot(bs_i, len(data.CHI_BINS), chi_4[:, None], use_cuda=use_cuda)

    y = torch.LongTensor(y)
    y_onehot = make_onehot(bs_i, 20, y[:, None], use_cuda=use_cuda)
    if use_cuda:
        y = y.cuda()

    # ensemble prediction over all models -- average logits
    logits_out = []
    chi_feat_out = []
    chi_1_out = []
    chi_2_out = []
    chi_3_out = []
    chi_4_out = []

    with torch.no_grad():
        for model in curr_models:
            feat, res_pred_logits, chi_1_pred, chi_2_pred, chi_3_pred, chi_4_pred = model.get_feat(
                voxels, y_onehot, chi_1_onehot[:, 1:], chi_2_onehot[:, 1:], chi_3_onehot[:, 1:]
            )
            logits_out.append(res_pred_logits[None])
            chi_feat_out.append(feat[None])
            chi_1_out.append(chi_1_pred[None])
            chi_2_out.append(chi_2_pred[None])
            chi_3_out.append(chi_3_pred[None])
            chi_4_out.append(chi_4_pred[None])

    logits_out = torch.cat(logits_out, 0).mean(0)
    chi_feat_out = torch.cat(chi_feat_out, 0).mean(0)
    chi_1_logits = torch.cat(chi_1_out, 0).mean(0)
    chi_2_logits = torch.cat(chi_2_out, 0).mean(0)
    chi_3_logits = torch.cat(chi_3_out, 0).mean(0)
    chi_4_logits = torch.cat(chi_4_out, 0).mean(0)

    chi_1 = chi_1 - 1
    chi_2 = chi_2 - 1
    chi_3 = chi_3 - 1
    chi_4 = chi_4 - 1

    if use_cuda:
        chi_1 = (chi_1).cuda()
        chi_2 = (chi_2).cuda()
        chi_3 = (chi_3).cuda()
        chi_4 = (chi_4).cuda()

    return (
        logits_out,
        chi_feat_out,
        y,
        chi_1_logits,
        chi_2_logits,
        chi_3_logits,
        chi_4_logits,
        chi_1,
        chi_2,
        chi_3,
        chi_4,
        chi_angles,
        chi_mask,
    )  


def get_energy_from_feat(
    models,
    logits,
    chi_feat,
    y,
    chi_1_logits,
    chi_2_logits,
    chi_3_logits,
    chi_4_logits,
    chi_1,
    chi_2,
    chi_3,
    chi_4,
    chi_angles,
    chi_mask,
    include_rotamer_probs=0,
    return_log_ps=0,
    use_cuda=True,
):
    # get residue log probs
    # energy, energy_per_res,
    log_p_per_res, log_p_mean = get_energy_from_logits(logits, y)

    # get rotamer log_probs
    chi_1_mask = torch.zeros_like(chi_1)
    chi_2_mask = torch.zeros_like(chi_2)
    chi_3_mask = torch.zeros_like(chi_3)
    chi_4_mask = torch.zeros_like(chi_4)

    if use_cuda:
        chi_1_mask = chi_1_mask.cuda()
        chi_2_mask = chi_2_mask.cuda()
        chi_3_mask = chi_3_mask.cuda()
        chi_4_mask = chi_4_mask.cuda()

    chi_1_mask[chi_1 < 0] = 1
    chi_2_mask[chi_2 < 0] = 1
    chi_3_mask[chi_3 < 0] = 1
    chi_4_mask[chi_4 < 0] = 1

    chi_1[chi_1 < 0] = 0
    chi_2[chi_2 < 0] = 0
    chi_3[chi_3 < 0] = 0
    chi_4[chi_4 < 0] = 0

    log_p_per_res_chi_1, log_p_per_res_chi_1_mean = get_energy_from_logits(chi_1_logits, chi_1, mask=chi_1_mask, baseline=1.3183412514892)
    log_p_per_res_chi_2, log_p_per_res_chi_2_mean = get_energy_from_logits(chi_2_logits, chi_2, mask=chi_2_mask, baseline=1.5970909799808386)
    log_p_per_res_chi_3, log_p_per_res_chi_3_mean = get_energy_from_logits(chi_3_logits, chi_3, mask=chi_3_mask, baseline=2.231545756901711)
    log_p_per_res_chi_4, log_p_per_res_chi_4_mean = get_energy_from_logits(chi_4_logits, chi_4, mask=chi_4_mask, baseline=2.084356748355477)

    if return_log_ps:
        return log_p_mean, log_p_per_res_chi_1_mean, log_p_per_res_chi_2_mean, log_p_per_res_chi_3_mean, log_p_per_res_chi_4_mean

    if include_rotamer_probs:
        # get per residue log probs (autoregressive)
        log_p_per_res = log_p_per_res + log_p_per_res_chi_1 + log_p_per_res_chi_2 + log_p_per_res_chi_3 + log_p_per_res_chi_4
        # optimize mean log prob across residues
        log_p_mean = log_p_per_res.mean()

    return log_p_per_res, log_p_mean


def get_energy(models, pose=None, pdb=None, chain="A", bb_only=0, return_chi=0, use_cuda=1, log_path="./", include_rotamer_probs=0):
    if pdb is not None:
        atom_coords, atom_data, residue_bb_index_list, res_data, res_label, chis = data.get_pdb_data(
            pdb[pdb.rfind("/") + 1 : -4], data_dir=pdb[: pdb.rfind("/")], skip_download=1, assembly=0
        )
    else:
        assert pose is not None, "need to specify pose to calc energy"
        pose.dump_pdb(log_path + "/" + "curr.pdb")
        atom_coords, atom_data, residue_bb_index_list, res_data, res_label, chis = data.get_pdb_data(
            "curr", data_dir=log_path, skip_download=1, assembly=0
        )

    # get residue and rotamer logits
    logits, chi_feat, y, chi_1_logits, chi_2_logits, chi_3_logits, chi_4_logits, chi_1, chi_2, chi_3, chi_4, chi_angles, chi_mask = get_conv_feat(
        models, atom_coords, atom_data, residue_bb_index_list, res_data, res_label, chis, bb_only=bb_only, return_chi=return_chi, use_cuda=use_cuda
    )

    # get model negative log probs (model energy) 
    log_p_per_res, log_p_mean = get_energy_from_feat(
        models,
        logits,
        chi_feat,
        y,
        chi_1_logits,
        chi_2_logits,
        chi_3_logits,
        chi_4_logits,
        chi_1,
        chi_2,
        chi_3,
        chi_4,
        chi_angles,
        chi_mask,
        include_rotamer_probs=include_rotamer_probs,
        use_cuda=use_cuda,
    )

    if return_chi:
        return res_label, log_p_per_res, log_p_mean, logits, chi_feat, chi_angles, chi_mask, [chi_1, chi_2, chi_3, chi_4]
    return res_label, log_p_per_res, log_p_mean, logits, chi_feat, chi_angles, chi_mask


def get_chi_init_feat(curr_models, feat, res_onehot):
    chi_feat_out = []
    with torch.no_grad():
        for model in curr_models:
            chi_feat = model.get_chi_init_feat(feat, res_onehot)
            chi_feat_out.append(chi_feat[None])
        chi_feat = torch.cat(chi_feat_out, 0).mean(0)
    return chi_feat


def get_chi_1_logits(curr_models, chi_feat):
    chi_1_pred_out = []
    with torch.no_grad():
        for model in curr_models:
            chi_1_pred = model.get_chi_1(chi_feat)
            chi_1_pred_out.append(chi_1_pred[None])
        chi_1_pred_out = torch.cat(chi_1_pred_out, 0).mean(0)
    return chi_1_pred_out


def get_chi_2_logits(curr_models, chi_feat, chi_1_onehot):
    chi_2_pred_out = []
    with torch.no_grad():
        for model in curr_models:
            chi_2_pred = model.get_chi_2(chi_feat, chi_1_onehot)
            chi_2_pred_out.append(chi_2_pred[None])
        chi_2_pred_out = torch.cat(chi_2_pred_out, 0).mean(0)
    return chi_2_pred_out


def get_chi_3_logits(curr_models, chi_feat, chi_1_onehot, chi_2_onehot):
    chi_3_pred_out = []
    with torch.no_grad():
        for model in curr_models:
            chi_3_pred = model.get_chi_3(chi_feat, chi_1_onehot, chi_2_onehot)
            chi_3_pred_out.append(chi_3_pred[None])
        chi_3_pred_out = torch.cat(chi_3_pred_out, 0).mean(0)
    return chi_3_pred_out


def get_chi_4_logits(curr_models, chi_feat, chi_1_onehot, chi_2_onehot, chi_3_onehot):
    chi_4_pred_out = []
    with torch.no_grad():
        for model in curr_models:
            chi_4_pred = model.get_chi_4(chi_feat, chi_1_onehot, chi_2_onehot, chi_3_onehot)
            chi_4_pred_out.append(chi_4_pred[None])
        chi_4_pred_out = torch.cat(chi_4_pred_out, 0).mean(0)
    return chi_4_pred_out


def sample_chi(chi_logits, use_cuda=True):
    # sample chi bin from predicted distribution
    chi_dist = Categorical(logits=chi_logits)
    chi_idx = chi_dist.sample().cpu().data.numpy()
    chi = torch.LongTensor(chi_idx) 
    # get one-hot encoding of sampled bin for autoregressive unroll
    chi_onehot = make_onehot(chi_logits.size()[0], len(data.CHI_BINS) - 1, chi[:, None], use_cuda=use_cuda)
    # sample chi angle (real) uniformly within bin
    chi_real = np.random.uniform(low=data.CHI_BINS[chi_idx], high=data.CHI_BINS[chi_idx + 1])
    return chi, chi_real, chi_onehot


def get_symm_chi(chi_pred_out, symm_idx_ptr, use_cuda=True):
    chi_pred_out_symm = []
    for i, ptr in enumerate(symm_idx_ptr):
        chi_pred_out_symm.append(chi_pred_out[ptr].mean(0)[None])
    chi_pred_out = torch.cat(chi_pred_out_symm, 0)
    chi, chi_real, chi_onehot = sample_chi(chi_pred_out, use_cuda=use_cuda)
    chi_real_out = []
    for i, ptr in enumerate(symm_idx_ptr):
        chi_real_out.append([chi_real[i][None] for j in range(len(ptr))])  # , 0))
    chi_real = np.concatenate(chi_real_out, axis=0)

    chi_onehot_out = []
    for i, ptr in enumerate(symm_idx_ptr):
        chi_onehot_out.append(torch.cat([chi_onehot[i][None] for j in range(len(ptr))], 0))
    chi_onehot = torch.cat(chi_onehot_out, 0)
    return chi_real, chi_onehot


# from https://codereview.stackexchange.com/questions/203319/greedy-graph-coloring-in-python
def color_nodes(graph, nodes):
    color_map = {}
    # Consider nodes in descending degree
    for node in nodes:  # sorted(graph, key=lambda x: len(graph[x]), reverse=True):
        neighbor_colors = set(color_map.get(neigh) for neigh in graph[node])
        color_map[node] = next(color for color in range(len(graph)) if color not in neighbor_colors)
    return color_map


################################
