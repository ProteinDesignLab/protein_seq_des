import torch

""" accuracy eval fns """

label_coarse = {0:0,1:0,2:0,   3:1,4:1,   5:2,6:2,   7:3,8:3,   9:4,10:4,11:4,12:4,13:4,  14:5,15:5,16:5,   17:6,18:7, 19:8}
label_res_single_dict_coarse= {0:'(+)', 1:'(-)', 2: 'ST', 3: 'NQ', 4: 'SH', 5:'LH', 6:'P', 7:'G', 8:'C'}

label_polar = {0:0, 1:0,2:0,3:0,4:0,5:0,6:0, 7:0,8:0,   9:1,10:1,11:1,12:1, 13:2,  14:1,  15:2,16:2,   17:3,18:4, 19:0}
label_res_single_dict_polar={0:'polar', 1: 'nonpolar', 2: 'amphipathic', 3: 'proline', 4:'glycine'}

def get_acc(logits, label, cm=None, label_dict=None, ignore_idx=None):

    pred = torch.argmax(logits, 1)

    if label_dict is not None:
        pred = torch.LongTensor([label_dict[p] for p in pred.cpu().data.numpy()])
        label = torch.LongTensor([label_dict[l] for l in label.cpu().data.numpy()])

    if ignore_idx is None:
        acc = float((pred == label).sum(-1)) / label.size()[0]
    else:
        if len(label[label != ignore_idx]) == 0:
            # case when all data in a batch is to be ignored
            acc = 0.0
        else:
            acc = float((pred[label != ignore_idx ] == label[label != ignore_idx]).sum(-1)) / len(label[label != ignore_idx]) 

    if cm is not None:
        if ignore_idx is None:
            for i in range(pred.size()[0]):
                # NOTE -- do not try to un-for loop this... errors
                cm[label[i], pred[i]] += 1

        else:
            for i in range(pred.size()[0]):
                # NOTE -- do not try to un-for loop this... errors
                if label[i] != ignore_idx:
                    cm[label[i], pred[i]] += 1

    return acc, cm


def get_chi_acc(logits, label, res_label, cm_dict=None, label_dict=None, ignore_idx=None):

    pred = torch.argmax(logits, 1)

    if label_dict is not None:
        pred = torch.LongTensor([label_dict[p] for p in pred.cpu().data.numpy()])
        label = torch.LongTensor([label_dict[l] for l in label.cpu().data.numpy()])

    if ignore_idx is None:
        acc = float((pred == label).sum(-1)) / label.size()[0]
    else:
        if len(label[label != ignore_idx]) == 0:
            # case when all data in a batch is to be ignored
            acc = 0.0
        else:
            acc = float((pred[label != ignore_idx ] == label[label != ignore_idx]).sum(-1)) / len(label[label != ignore_idx])

    if cm_dict is not None:
        if ignore_idx is None:
            for i in range(pred.size()[0]):
                # NOTE -- do not try to un-for loop this... errors
                cm_dict[res_label[i].item()][label[i], pred[i]] += 1

        else:
            for i in range(pred.size()[0]):
                # NOTE -- do not try to un-for loop this... errors
                if label[i] != ignore_idx:
                    cm_dict[res_label[i].item()][label[i], pred[i]] += 1

    return acc, cm_dict


def get_chi_EV(probs, label, res_label, cm_dict=None, label_dict=None, ignore_idx=None):


    if cm_dict is not None:
        if ignore_idx is None:
            for i in range(probs.shape[0]): #ize()[0]):
                # NOTE -- do not try to un-for loop this... errors
                cm_dict[res_label[i].item()]['ev'] += probs[i]
                cm_dict[res_label[i].item()]['n']+= 1

        else:
            for i in range(probs.shape[0]): #ize()[0]):
                # NOTE -- do not try to un-for loop this... errors
                if label[i] != ignore_idx:
                    cm_dict[res_label[i].item()]['ev'] += probs[i]
                    cm_dict[res_label[i].item()]['n']+= 1

    return cm_dict


# from pytorch ...
def get_top_k_acc(output, target, k=3, ignore_idx=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        if ignore_idx is not None:
            pred = pred[target !=ignore_idx]
            target = target[target !=ignore_idx]

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct = correct.contiguous()
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #res.append(correct_k.mul_(100.0 / batch_size))
        return correct_k.mul_(1.0 / batch_size).item()

###



