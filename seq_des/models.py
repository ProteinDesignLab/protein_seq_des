import torch
import torch.nn as nn
import seq_des.util.data as data
import common.atoms


def init_ortho_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Conv2d):
            torch.nn.init.orthogonal_(module.weight)
        elif isinstance(module, nn.ConvTranspose2d):
            torch.nn.init.orthogonal_(module.weight)


class seqPred(nn.Module):
    def __init__(self, nic, nf=64, momentum=0.01):
        super(seqPred, self).__init__()
        self.nic = nic
        self.model = nn.Sequential(
            # 20 -- 10
            nn.Conv3d(nic, nf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(nf, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(nf, nf, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(nf, nf, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            # 10 -- 5
            nn.Conv3d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(nf * 2, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(nf * 2, nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf * 2, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(nf * 2, nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf * 2, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            # 5 -- 1
            nn.Conv3d(nf * 2, nf * 4, 5, 1, 0, bias=False),
            nn.BatchNorm3d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
        )

        # res pred
        self.out = nn.Sequential(
            nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(nf * 4, len(common.atoms.label_res_dict.keys()), 3, 1, 1, bias=False),
        )

        # chi feat vec -- condition on residue and env feature vector
        self.chi_feat = nn.Sequential(
            nn.Conv1d(nf * 4 + 20, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
        )

        # chi 1 pred -- condition on chi feat vec
        self.chi_1_out = nn.Sequential(
            nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(nf * 4, (len(data.CHI_BINS) - 1), 3, 1, 1, bias=False),
        )

        # chi 2 pred -- condition on chi 1 and chi feat vec
        self.chi_2_out = nn.Sequential(
            nn.Conv1d(nf * 4 + 1 * (len(data.CHI_BINS) - 1), nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(nf * 4, (len(data.CHI_BINS) - 1), 3, 1, 1, bias=False),
        )

        # chi 3 pred -- condition on chi 1, chi 2, and chi feat vec
        self.chi_3_out = nn.Sequential(
            nn.Conv1d(nf * 4 + 2 * (len(data.CHI_BINS) - 1), nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(nf * 4, (len(data.CHI_BINS) - 1), 3, 1, 1, bias=False),
        )

        # chi 4 pred -- condition on chi 1, chi 2, chi 3, and chi feat vec
        self.chi_4_out = nn.Sequential(
            nn.Conv1d(nf * 4 + 3 * (len(data.CHI_BINS) - 1), nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(nf * 4, (len(data.CHI_BINS) - 1), 3, 1, 1, bias=False),
        )

    def res_pred(self, input):
        bs = input.size()[0]
        feat = self.model(input).view(bs, -1, 1)
        res_pred = self.out(feat).view(bs, -1)
        return res_pred, feat

    def get_chi_init_feat(self, feat, res_onehot):
        chi_init = torch.cat([feat, res_onehot[..., None]], 1)
        chi_feat = self.chi_feat(chi_init)
        return chi_feat

    def get_chi_1(self, chi_feat):
        chi_1_pred = self.chi_1_out(chi_feat).view(chi_feat.size()[0], -1)
        return chi_1_pred

    def get_chi_2(self, chi_feat, chi_1_onehot):
        chi_2_pred = self.chi_2_out(torch.cat([chi_feat, chi_1_onehot[..., None]], 1)).view(chi_feat.size()[0], -1)
        return chi_2_pred

    def get_chi_3(self, chi_feat, chi_1_onehot, chi_2_onehot):
        chi_3_pred = self.chi_3_out(torch.cat([chi_feat, chi_1_onehot[..., None], chi_2_onehot[..., None]], 1)).view(chi_feat.size()[0], -1)
        return chi_3_pred

    def get_chi_4(self, chi_feat, chi_1_onehot, chi_2_onehot, chi_3_onehot):
        chi_4_pred = self.chi_4_out(torch.cat([chi_feat, chi_1_onehot[..., None], chi_2_onehot[..., None], chi_3_onehot[..., None]], 1)).view(
            chi_feat.size()[0], -1
        )
        return chi_4_pred

    def get_feat(self, input, res_onehot, chi_1_onehot, chi_2_onehot, chi_3_onehot):
        bs = input.size()[0]
        feat = self.model(input).view(bs, -1, 1)
        res_pred = self.out(feat).view(bs, -1)

        # condition on res type and env feat
        chi_init = torch.cat([feat, res_onehot[..., None]], 1)
        chi_feat = self.chi_feat(chi_init)

        # condition on true residue type and previous ground-truth rotamer angles
        chi_1_pred = self.chi_1_out(chi_feat).view(bs, -1)
        chi_2_pred = self.chi_2_out(torch.cat([chi_feat, chi_1_onehot[..., None]], 1)).view(bs, -1)
        chi_3_pred = self.chi_3_out(torch.cat([chi_feat, chi_1_onehot[..., None], chi_2_onehot[..., None]], 1)).view(bs, -1)
        chi_4_pred = self.chi_4_out(torch.cat([chi_feat, chi_1_onehot[..., None], chi_2_onehot[..., None], chi_3_onehot[..., None]], 1)).view(bs, -1)
        return feat, res_pred, chi_1_pred, chi_2_pred, chi_3_pred, chi_4_pred


    def forward(self, input, res_onehot, chi_1_onehot, chi_2_onehot, chi_3_onehot):
        bs = input.size()[0]
        feat = self.model(input).view(bs, -1, 1)
        res_pred = self.out(feat).view(bs, -1)

        # condition on res type and env feat
        chi_init = torch.cat([feat, res_onehot[..., None]], 1)
        chi_feat = self.chi_feat(chi_init)

        # condition on true residue type and previous ground-truth rotamer angles
        chi_1_pred = self.chi_1_out(chi_feat).view(bs, -1)
        chi_2_pred = self.chi_2_out(torch.cat([chi_feat, chi_1_onehot[..., None]], 1)).view(bs, -1)
        chi_3_pred = self.chi_3_out(torch.cat([chi_feat, chi_1_onehot[..., None], chi_2_onehot[..., None]], 1)).view(bs, -1)
        chi_4_pred = self.chi_4_out(torch.cat([chi_feat, chi_1_onehot[..., None], chi_2_onehot[..., None], chi_3_onehot[..., None]], 1)).view(bs, -1)

        return res_pred, chi_1_pred, chi_2_pred, chi_3_pred, chi_4_pred
