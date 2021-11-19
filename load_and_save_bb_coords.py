import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import common.run_manager
import glob
import seq_des.util.canonicalize as canonicalize
import pickle
import seq_des.util.data as datasets
from torch.utils import data


import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

""" script to load PDB coords, canonicalize, save """

def main():

    manager = common.run_manager.RunManager()

    manager.parse_args()
    args = manager.args
    log = manager.log

    dataset = datasets.PDB_domain_spitter(txt_file=args.txt, pdb_path=args.pdb_dir, num_return=75, bb_only=1)

    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.workers)

    num_return = args.num_return
    gen = iter(dataloader)
    coords_out, data_out, ys, domain_ids, chis_out = [], [], [], [], []

    cs = args.chunk_size
    n = 0

    for it in tqdm(range(len(dataloader)), desc="loading and saving coords"):

        out = gen.next()
        if len(out) == 0 or out is None:
            print("out is none")
            continue
        atom_coords, atom_data, res_label, domain_id, chis = out
        for i in range(len(atom_coords)):
            coords_out.extend(atom_coords[i][0].cpu().data.numpy())
            data_out.extend(atom_data[i][0].cpu().data.numpy())
            ys.extend(res_label[i][0].cpu().data.numpy())
            domain_ids.extend([domain_id[i][0]] * res_label[i][0].cpu().data.numpy().shape[0])
            chis_out.extend(chis[i][0].cpu().data.numpy())

            assert len(coords_out) == len(ys)
            assert len(coords_out) == len(data_out)
            assert len(coords_out) == len(domain_ids), (len(coords_out), len(domain_ids))
            assert len(coords_out) == len(chis_out)

        del atom_coords
        del atom_data
        del res_label
        del domain_id

        # intermittent save data
        if len(coords_out) > cs or it == len(dataloader) - 1:
            # shuffle then save
            print(n, len(coords_out))  # -- NOTE keep this
            idx = np.arange(min(cs, len(coords_out)))
            np.random.shuffle(idx)
            print(n, len(idx))

            c, d, y, di, ch = map(lambda arr: np.array(arr[: len(idx)])[idx], [coords_out, data_out, ys, domain_ids, chis_out])

            print("saving", args.save_dir + "/" + "data_%0.4d.pt" % (n))
            torch.save((c, d, y, di, ch), args.save_dir + "/" + "data_%0.4d.pt" % (n))

            print("Current num examples", (n) * cs + len(coords_out))

            n += 1
            coords_out, data_out, ys, domain_ids, chis_out = map(lambda arr: arr[len(idx) :], [coords_out, data_out, ys, domain_ids, chis_out])


if __name__ == "__main__":
    main()
