import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses


def hsic_loss(z1, z2):

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    B = z1.size(0)

    z1 = z1 - z1.mean(0)
    z2 = z2 - z2.mean(0)

    K = z1 @ z1.t()
    L = z2 @ z2.t()

    H = torch.eye(B, device=z1.device) - 1.0 / B

    Kc = H @ K @ H
    Lc = H @ L @ H

    return (Kc * Lc).sum() / ((B - 1) ** 2)

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha


    def proxy_anchor(self, X, T):
        P = self.proxies
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term
        return loss

    def forward(self, result, T):
        cls_fea = result['cls_fea']
        metric_loss = self.proxy_anchor(cls_fea, T)

        #recon_loss = result['recon_loss']  # ~ 0.1 - 1
        kl_loss = result['kl_loss']  # ~ 1 - 10
        cross_loss = result['cross_loss']  # ~ 1e-3 - 1e-2（非常小！）
        cluster_loss = result['cluster_loss']
        diversity_loss = result['diversity_loss']

        orth_loss = result['orth_loss']
        loss = (
                metric_loss
                + result['cls_loss']
                + 1000*orth_loss
                + 10 * cross_loss
                + 50*cluster_loss
                + 100*diversity_loss
                + 1e-6  * kl_loss
        )

        return loss, [
            metric_loss,
            result['cls_loss'],
            1000*orth_loss,
            10*cross_loss,
            50*cluster_loss,
            100*diversity_loss,
            1e-6* kl_loss
        ]


