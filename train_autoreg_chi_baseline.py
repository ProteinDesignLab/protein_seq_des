import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import common.run_manager
import seq_des.models as models
import seq_des.util.voxelize as voxelize
import glob
import seq_des.util.canonicalize as canonicalize
import pickle
import seq_des.util.data as datasets
from torch.utils import data
import common.atoms
import seq_des.util.acc_util as acc_util
import subprocess as sp
import time
import torch.nn.functional as F

""" script to train 3D CNN on local residue-centered environments -- BB only -- with autoregressive rotamer chi angle prediction"""

dist = 10
n = 20
c = len(common.atoms.atoms)


def test(
    model, gen, dataloader, criterion, chi_1_criterion, chi_2_criterion, chi_3_criterion, chi_4_criterion, max_it=1e6, desc="test", device=0, batch_size=64, n_iters=500, k=3, use_cuda=True,
):
    n_iters = min(max_it, n_iters)
    model = model.eval()
    gen = iter(dataloader)
    (losses, avg_acc, avg_top_k_acc, avg_coarse_acc, avg_polar_acc, avg_chi_1_acc, avg_chi_2_acc, avg_chi_3_acc, avg_chi_4_acc, avg_chi_1_loss, avg_chi_2_loss, avg_chi_3_loss, avg_chi_4_loss,) = ([] for i in range(13))
    with torch.no_grad():

        for i in tqdm(range(n_iters), desc=desc):
            try:
                out = gen.next()
            except StopIteration:
                gen = iter(dataloader)
                out = gen.next()

            out = step(model, out, criterion, chi_1_criterion, chi_2_criterion, chi_3_criterion, chi_4_criterion, use_cuda=use_cuda)

            if out is None:
                continue
            (loss, chi_1_loss, chi_2_loss, chi_3_loss, chi_4_loss, out, y, acc, top_k_acc, coarse_acc, polar_acc, chi_1_acc, chi_2_acc, chi_3_acc, chi_4_acc,) = out

            # append losses, accs to lists
            for x, y in zip(
                [losses, avg_acc, avg_top_k_acc, avg_coarse_acc, avg_polar_acc, avg_chi_1_acc, avg_chi_2_acc, avg_chi_3_acc, avg_chi_4_acc, avg_chi_1_loss, avg_chi_2_loss, avg_chi_3_loss, avg_chi_4_loss,],
                [loss.item(), acc, top_k_acc, coarse_acc, polar_acc, chi_1_acc, chi_2_acc, chi_3_acc, chi_4_acc, chi_1_loss.item(), chi_2_loss.item(), chi_3_loss.item(), chi_4_loss.item(),],
            ):
                x.append(y)

            del (
                loss,
                chi_1_loss,
                chi_2_loss,
                chi_3_loss,
                chi_4_loss,
                out,
                y,
                acc,
                top_k_acc,
                coarse_acc,
                polar_acc,
                chi_1_acc,
                chi_2_acc,
                chi_3_acc,
                chi_4_acc,
            )

        print(
            "\nloss", np.mean(losses), "acc", np.mean(avg_acc), "top3", np.mean(avg_top_k_acc), "coarse", np.mean(avg_coarse_acc), "polar", np.mean(avg_polar_acc),
        )

    return (
        gen,
        np.mean(losses),
        np.mean(avg_chi_1_loss),
        np.mean(avg_chi_2_loss),
        np.mean(avg_chi_3_loss),
        np.mean(avg_chi_4_loss),
        np.mean(avg_acc),
        np.mean(avg_top_k_acc),
        np.mean(avg_coarse_acc),
        np.mean(avg_polar_acc),
        np.mean(avg_chi_1_acc),
        np.mean(avg_chi_2_acc),
        np.mean(avg_chi_3_acc),
        np.mean(avg_chi_4_acc),
    )


def step(model, out, criterion, chi_1_criterion, chi_2_criterion, chi_3_criterion, chi_4_criterion, k=3, use_cuda=True):

    (bs_idx, x_atom, x_bb, x_b, y_b, z_b, x_res_type, y, chi_angles_real, chi_angles,) = out

    bs = len(bs_idx)
    output_atom = torch.zeros((bs, c + 1, n + 2, n + 2, n + 2)).cuda()
    output_atom[bs_idx, x_atom, x_b, y_b, z_b] = 1  # atom type
    X = output_atom[:, :c, 1:-1, 1:-1, 1:-1]

    if X is None:
        return None

    X, y = X.float(), y.long()
    chi_angles = chi_angles.long()

    chi_1 = chi_angles[:, 0]
    chi_2 = chi_angles[:, 1]
    chi_3 = chi_angles[:, 2]
    chi_4 = chi_angles[:, 3]

    y_onehot = torch.FloatTensor(y.size()[0], 20)
    y_onehot.zero_()
    y_onehot.scatter_(1, y[:, None], 1)

    chi_1_onehot = torch.FloatTensor(chi_1.size()[0], len(datasets.CHI_BINS))
    chi_1_onehot.zero_()
    chi_1_onehot.scatter_(1, chi_1[:, None], 1)

    chi_2_onehot = torch.FloatTensor(chi_2.size()[0], len(datasets.CHI_BINS))
    chi_2_onehot.zero_()
    chi_2_onehot.scatter_(1, chi_2[:, None], 1)

    chi_3_onehot = torch.FloatTensor(chi_3.size()[0], len(datasets.CHI_BINS))
    chi_3_onehot.zero_()
    chi_3_onehot.scatter_(1, chi_3[:, None], 1)

    if use_cuda:
        (X, y, y_onehot, chi_1_onehot, chi_2_onehot, chi_3_onehot, chi_1, chi_2, chi_3, chi_4,) = map(lambda x: x.cuda(), [X, y, y_onehot, chi_1_onehot, chi_2_onehot, chi_3_onehot, chi_1, chi_2, chi_3, chi_4,],)

    out, chi_1_pred, chi_2_pred, chi_3_pred, chi_4_pred = model(X, y_onehot, chi_1_onehot[:, 1:], chi_2_onehot[:, 1:], chi_3_onehot[:, 1:])
    # loss
    loss = criterion(out, y)
    chi_1_loss = chi_1_criterion(chi_1_pred, chi_1 - 1)
    chi_2_loss = chi_2_criterion(chi_2_pred, chi_2 - 1)
    chi_3_loss = chi_3_criterion(chi_3_pred, chi_3 - 1)
    chi_4_loss = chi_4_criterion(chi_4_pred, chi_4 - 1)

    # acc
    acc, _ = acc_util.get_acc(out, y)
    top_k_acc = acc_util.get_top_k_acc(out, y, k=k)
    coarse_acc, _ = acc_util.get_acc(out, y, label_dict=acc_util.label_coarse)
    polar_acc, _ = acc_util.get_acc(out, y, label_dict=acc_util.label_polar)
    chi_1_acc, _ = acc_util.get_acc(chi_1_pred, chi_1 - 1, ignore_idx=-1)
    chi_2_acc, _ = acc_util.get_acc(chi_2_pred, chi_2 - 1, ignore_idx=-1)
    chi_3_acc, _ = acc_util.get_acc(chi_3_pred, chi_3 - 1, ignore_idx=-1)
    chi_4_acc, _ = acc_util.get_acc(chi_4_pred, chi_4 - 1, ignore_idx=-1)

    return (
        loss,
        chi_1_loss,
        chi_2_loss,
        chi_3_loss,
        chi_4_loss,
        out,
        y,
        acc,
        top_k_acc,
        coarse_acc,
        polar_acc,
        chi_1_acc,
        chi_2_acc,
        chi_3_acc,
        chi_4_acc,
    )


def step_iter(gen, dataloader):
    try:
        out = gen.next()
    except StopIteration:
        gen = iter(dataloader)
        out = gen.next()
    return gen, out


def main():

    manager = common.run_manager.RunManager()

    manager.parse_args()
    args = manager.args
    log = manager.log

    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # set up model
    model = models.seqPred(nic=len(common.atoms.atoms), nf=args.nf, momentum=args.momentum)
    model.apply(models.init_ortho_weights)
    if use_cuda:
        model.cuda()
    else:
        print("Training model on CPU")
    print(model)

    # parallelize over available GPUs
    if torch.cuda.device_count() > 1:
        print("using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    if args.model != "":
        # load pretrained model
        model.load_state_dict(torch.load(args.model))
        print("loaded pretrained model")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.reg)

    if args.optimizer != "":
        # load pretrained optimizer
        optimizer.load_state_dict(torch.load(args.optimizer))
        print("loaded pretrained optimizer")

    # load pretrained model weights / optimizer state

    chi_1_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    chi_2_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    chi_3_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    chi_4_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion.cuda()
        chi_1_criterion.cuda()
        chi_2_criterion.cuda()
        chi_3_criterion.cuda()
        chi_4_criterion.cuda()

    train_dataset = datasets.PDB_data_spitter(data_dir=args.data_dir + "/train_s95_chi_bb")
    train_dataset.len = 8145448  # NOTE -- need to update this if underlying data changes

    test_dataset = datasets.PDB_data_spitter(data_dir=args.data_dir + "/test_s95_chi_bb")
    test_dataset.len = 574267  # NOTE -- need to update this if underlying data changes

    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=datasets.collate_wrapper,)
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=datasets.collate_wrapper,)

    # training params
    validation_frequency = args.validation_frequency
    save_frequency = args.save_frequency

    """ TRAIN """

    model.train()
    gen = iter(train_dataloader)
    test_gen = iter(test_dataloader)
    bs = torch.cuda.device_count() * args.batchSize
    output_atom = torch.zeros((bs, c + 1, n + 2, n + 2, n + 2)).cuda()
    y_onehot = torch.FloatTensor(bs, 20).cuda()
    chi_1_onehot = torch.FloatTensor(bs, len(datasets.CHI_BINS)).cuda()
    chi_2_onehot = torch.FloatTensor(bs, len(datasets.CHI_BINS)).cuda()
    chi_3_onehot = torch.FloatTensor(bs, len(datasets.CHI_BINS)).cuda()

    for epoch in range(args.epochs):
        for it in tqdm(range(len(train_dataloader)), desc="training epoch %0.2d" % epoch):

            gen, out = step_iter(gen, train_dataloader)
            (bs_idx, x_atom, x_bb, x_b, y_b, z_b, x_res_type, y, chi_angles_real, chi_angles,) = out
            bs_i = len(bs_idx)
            output_atom.zero_()
            output_atom[bs_idx, x_atom, x_b, y_b, z_b] = 1  # atom type
            X = output_atom[:, :c, 1:-1, 1:-1, 1:-1]

            X, y = X.float(), y.long()
            chi_angles = chi_angles.long()

            chi_1 = chi_angles[:, 0]
            chi_2 = chi_angles[:, 1]
            chi_3 = chi_angles[:, 2]
            chi_4 = chi_angles[:, 3]

            if use_cuda:
                y, y_onehot, chi_1, chi_2, chi_3, chi_4 = map(lambda x: x.cuda(), [y, y_onehot, chi_1, chi_2, chi_3, chi_4])

            if bs_i < bs:
                y = F.pad(y, (0, bs - bs_i))
                chi_1 = F.pad(chi_1, (0, bs - bs_i))
                chi_2 = F.pad(chi_2, (0, bs - bs_i))
                chi_3 = F.pad(chi_3, (0, bs - bs_i))

            y_onehot.zero_()
            y_onehot.scatter_(1, y[:, None], 1)

            chi_1_onehot.zero_()
            chi_1_onehot.scatter_(1, chi_1[:, None], 1)

            chi_2_onehot.zero_()
            chi_2_onehot.scatter_(1, chi_2[:, None], 1)

            chi_3_onehot.zero_()
            chi_3_onehot.scatter_(1, chi_3[:, None], 1)

            out, chi_1_pred, chi_2_pred, chi_3_pred, chi_4_pred = model(X[:bs_i], y_onehot[:bs_i], chi_1_onehot[:bs_i, 1:], chi_2_onehot[:bs_i, 1:], chi_3_onehot[:bs_i, 1:])
            res_loss = criterion(out, y[:bs_i])
            chi_1_loss = chi_1_criterion(chi_1_pred, chi_1[:bs_i] - 1)
            chi_2_loss = chi_2_criterion(chi_2_pred, chi_2[:bs_i] - 1)
            chi_3_loss = chi_3_criterion(chi_3_pred, chi_3[:bs_i] - 1)
            chi_4_loss = chi_4_criterion(chi_4_pred, chi_4[:bs_i] - 1)

            train_loss = res_loss + chi_1_loss + chi_2_loss + chi_3_loss + chi_4_loss
            train_loss.backward()
            optimizer.step()

            # acc
            train_acc, _ = acc_util.get_acc(out, y[:bs_i], cm=None)
            train_top_k_acc = acc_util.get_top_k_acc(out, y[:bs_i], k=3)
            train_coarse_acc, _ = acc_util.get_acc(out, y[:bs_i], label_dict=acc_util.label_coarse)
            train_polar_acc, _ = acc_util.get_acc(out, y[:bs_i], label_dict=acc_util.label_polar)

            chi_1_acc, _ = acc_util.get_acc(chi_1_pred, chi_1[:bs_i] - 1, ignore_idx=-1)
            chi_2_acc, _ = acc_util.get_acc(chi_2_pred, chi_2[:bs_i] - 1, ignore_idx=-1)
            chi_3_acc, _ = acc_util.get_acc(chi_3_pred, chi_3[:bs_i] - 1, ignore_idx=-1)
            chi_4_acc, _ = acc_util.get_acc(chi_4_pred, chi_4[:bs_i] - 1, ignore_idx=-1)

            # tensorboard logging
            map(
                lambda x: log.log_scalar("seq_chi_pred/%s" % x[0], x[1]),
                zip(
                    ["res_loss", "chi_1_loss", "chi_2_loss", "chi_3_loss", "chi_4_loss", "train_acc", "chi_1_acc", "chi_2_acc", "chi_3_acc", "chi_4_acc", "train_top3_acc", "train_coarse_acc", "train_polar_acc",],
                    [res_loss.item(), chi_1_loss.item(), chi_2_loss.item(), chi_3_loss.item(), chi_4_loss.item(), train_acc, chi_1_acc, chi_2_acc, chi_3_acc, chi_4_acc, train_top_k_acc, train_coarse_acc, train_polar_acc,],
                ),
            )

            if it % validation_frequency == 0 or it == len(train_dataloader) - 1:

                if it > 0:
                    if torch.cuda.device_count() > 1:
                        torch.save(
                            model.module.state_dict(), log.log_path + "/seq_chi_pred_baseline_curr_weights.pt",
                        )
                    else:
                        torch.save(
                            model.state_dict(), log.log_path + "/seq_chi_pred_baseline_curr_weights.pt",
                        )
                    torch.save(
                        optimizer.state_dict(), log.log_path + "/seq_chi_pred_baseline_curr_optimizer.pt",
                    )

                # NOTE -- saving models for each validation step
                if it > 0 and (it % save_frequency == 0 or it == len(train_dataloader) - 1):
                    if torch.cuda.device_count() > 1:
                        torch.save(
                            model.module.state_dict(), log.log_path + "/seq_chi_pred_baseline_epoch_%0.3d_%s_weights.pt" % (epoch, it),
                        )
                    else:
                        torch.save(
                            model.state_dict(), log.log_path + "/seq_chi_pred_baseline_epoch_%0.3d_%s_weights.pt" % (epoch, it),
                        )

                    torch.save(
                        optimizer.state_dict(), log.log_path + "/seq_chi_pred_baseline_epoch_%0.3d_%s_optimizer.pt" % (epoch, it),
                    )

                ##NOTE -- turning back on model.eval()
                model.eval()
                # eval on the test set
                (test_gen, curr_test_loss, test_chi_1_loss, test_chi_2_loss, test_chi_3_loss, test_chi_4_loss, curr_test_acc, curr_test_top_k_acc, coarse_acc, polar_acc, chi_1_acc, chi_2_acc, chi_3_acc, chi_4_acc,) = test(
                    model,
                    test_gen,
                    test_dataloader,
                    criterion,
                    chi_1_criterion,
                    chi_2_criterion,
                    chi_3_criterion,
                    chi_4_criterion,
                    max_it=len(test_dataloader),
                    n_iters=min(10, len(test_dataloader)),
                    desc="test",
                    device=device,
                    batch_size=args.batchSize,
                    use_cuda=use_cuda,
                )

                map(
                    lambda x: log.log_scalar("seq_chi_pred/%s" % x[0], x[1]),
                    zip(
                        [
                            "test_loss",
                            "test_chi_1_loss",
                            "test_chi_2_loss",
                            "test_chi_3_loss",
                            "test_chi_4_loss",
                            "test_acc",
                            "test_chi_1_acc",
                            "test_chi_2_acc",
                            "test_chi_3_acc",
                            "test_chi_4_acc",
                            "test_acc_top3",
                            "test_coarse_acc",
                            "test_polar_acc",
                        ],
                        [
                            curr_test_loss.item(),
                            chi_1_loss.item(),
                            chi_2_loss.item(),
                            chi_3_loss.item(),
                            chi_4_loss.item(),
                            curr_test_acc.item(),
                            chi_1_acc.item(),
                            chi_2_acc.item(),
                            chi_3_acc.item(),
                            chi_4_acc.item(),
                            curr_test_top_k_acc.item(),
                            coarse_acc.item(),
                            polar_acc.item(),
                        ],
                    ),
                )

                model.train()

            log.advance_iteration()


if __name__ == "__main__":
    main()
