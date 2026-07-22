import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import MultivariateNormalDiag
from torch.autograd import Function

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50,ResNet50_Weights
from torchvision.models import resnet101
import torch.utils.model_zoo as model_zoo

class Resnet50(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=True, bn_freeze = True):
        super(Resnet50, self).__init__()

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.embedding_size = embedding_size
        self.nb_classes = 100
        self.nb_clusters = 20
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding_cls = nn.Linear(self.num_ftrs, self.embedding_size)
        self.model.embedding_var = nn.Linear(self.num_ftrs, self.embedding_size)

        self.model.mu_layer_cls = nn.Linear(self.embedding_size, self.embedding_size)
        self.model.logvar_layer_cls = nn.Linear(self.embedding_size, self.embedding_size)

        self.model.mu_layer_var = nn.Linear(self.embedding_size, self.embedding_size)
        self.model.logvar_layer_var = nn.Linear(self.embedding_size, self.embedding_size)

        self.model.cls_centers = nn.Parameter(torch.empty(self.nb_clusters,self.embedding_size))
        nn.init.kaiming_normal_(self.model.cls_centers, mode='fan_out')


        self.model.decoder = nn.Sequential(
            nn.Linear(self.embedding_size * 2, self.embedding_size*4),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_size *4, self.num_ftrs)
        )



        self._initialize_weights()
        self._initialize_decoder_weights()

        self.criterionL1 = nn.L1Loss()

        self.model.cls_head = nn.Linear(self.embedding_size, self.nb_classes)
        nn.init.kaiming_normal_(self.model.cls_head.weight, mode='fan_out')
        nn.init.constant_(self.model.cls_head.bias, 0)

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def purification_loss(self, z_star, z_prime, labels):
        device = z_star.device
        B = z_star.size(0)

        idx = torch.arange(B, device=device)
        i, j = idx.repeat_interleave(B), idx.repeat(B)

        mask = (i != j) & (labels[i] == labels[j])
        i, j = i[mask], j[mask]

        if not i.numel():
            return torch.tensor(0.0, device=device)

        z_mix = torch.cat([z_star[i], z_prime[j]], dim=1)
        z_mix2 = torch.cat([z_star[j], z_prime[i]], dim=1)


        recon_feature = self.model.decoder(z_mix)
        recon_features2 = self.model.decoder(z_mix2)

        recon_cls_fea = self.model.embedding_cls(recon_feature)
        recon_cls_fea2 = self.model.embedding_cls(recon_features2)

        recon_mu = self.model.mu_layer_cls(recon_cls_fea)
        recon_mu2 = self.model.mu_layer_cls(recon_cls_fea2)

        recon_mu = F.normalize(recon_mu,dim=1)
        recon_mu2 = F.normalize(recon_mu2,dim=1)

        target = F.normalize(z_star[i].detach(), dim=1)
        target2 = F.normalize(z_star[j].detach(),dim=1)

        return (1 - (recon_mu * target).sum(dim=1)).mean() + (1-(recon_mu2*target2).sum(dim=1)).mean()



    def balanced_anchor_loss(self, feats, anchors, tau=0.1):
        feats = F.normalize(feats, dim=1)
        anchors = F.normalize(anchors, dim=1)
        sim = feats @ anchors.T
        q = torch.softmax(sim / tau, dim=1)
        recon = q @ anchors
        recon = F.normalize(recon, dim=1)
        loss = (1 - F.cosine_similarity(feats, recon, dim=1)).mean()
        c_sim = anchors @ anchors.T
        mask = 1 - torch.eye(anchors.size(0),device=anchors.device)
        diversity = ((c_sim * mask).pow(2).sum()/ mask.sum())
        return loss, diversity, q

    def hsic(self, z_rel, z_var):

        z_rel = F.normalize(z_rel, dim=1)
        z_var = F.normalize(z_var, dim=1)

        B = z_rel.size(0)

        K = z_rel @ z_rel.t()
        L = z_var @ z_var.t()

        H = torch.eye(B, device=K.device) - torch.ones(B, B, device=K.device) / B

        Kc = H @ K @ H
        Lc = H @ L @ H

        loss = (Kc * Lc).mean()

        return loss

    def forward(self, x, T=None, epoch=None):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        avg = self.model.gap(x)
        max = self.model.gmp(x)
        x = avg + max
        x = x.view(x.size(0),-1)
        x = F.layer_norm(x, (self.num_ftrs,))


        cls_fea = self.model.embedding_cls(x)
        cls_fea_ln = F.layer_norm(cls_fea,(self.embedding_size,))

        cls_fea_norm = F.normalize(cls_fea, dim=1)
        if T is None:
            return {
                'cls_fea': cls_fea_norm
            }


        mu_cls = self.model.mu_layer_cls(cls_fea_ln)
        logvar_cls = self.model.logvar_layer_cls(cls_fea_ln)
        std_cls = torch.exp(0.5 * logvar_cls)
        eps_cls = torch.randn_like(std_cls)
        z_rel = mu_cls + eps_cls * std_cls
        kl_loss_cls = 0.5 * torch.sum(mu_cls ** 2 + torch.exp(logvar_cls) - logvar_cls - 1.0, dim=1).mean()

        logits = self.model.cls_head(mu_cls)
        cls_loss = F.cross_entropy(logits, T)


        var_fea = self.model.embedding_var(x)
        var_fea_ln = F.layer_norm(var_fea, (self.embedding_size,))
        cluster_loss, diversity_loss,q = self.balanced_anchor_loss(var_fea,anchors=self.model.cls_centers)
        mu_var = self.model.mu_layer_var(var_fea_ln)
        logvar_var = self.model.logvar_layer_var(var_fea_ln)

        std_var = torch.exp(0.5 * logvar_var)
        eps_var = torch.randn_like(std_var)
        z_var = mu_var + eps_var * std_var
        #z_joint = torch.cat([z_rel, z_var], dim=1)

        #x_recon = self.model.decoder(z_joint)
        #x_recon = F.layer_norm(x_recon,(self.num_ftrs,))
        #recon_loss = self.criterionL1(x_recon, x)

        kl_loss_var = 0.5 * torch.sum(mu_var  ** 2 + torch.exp(logvar_var)- logvar_var- 1.0, dim=1).mean()

        kl_loss = kl_loss_cls + kl_loss_var

        orth_loss = self.hsic(z_rel,z_var)
        cross_loss = self.purification_loss(z_rel,  z_var, T)

        return {
            'cls_fea':cls_fea,
            'var_fea':var_fea,
            'cls_loss':cls_loss,
            'z_rel':z_rel,
            'z_var':z_var,
            #'recon_loss':recon_loss,
            'kl_loss':kl_loss,
            'orth_loss':orth_loss,
            'cross_loss':cross_loss,
            'cluster_loss':cluster_loss,
            'diversity_loss':diversity_loss
        }



    def _initialize_weights(self):
        # embedding
        init.kaiming_normal_(self.model.embedding_cls.weight, mode='fan_out')
        init.constant_(self.model.embedding_cls.bias, 0)
        init.kaiming_normal_(self.model.embedding_var.weight, mode='fan_out')
        init.constant_(self.model.embedding_var.bias, 0)

        # cls vae
        nn.init.kaiming_normal_(self.model.mu_layer_cls.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.model.mu_layer_cls.bias, 0.0)
        nn.init.kaiming_normal_(self.model.logvar_layer_cls.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.model.logvar_layer_cls.bias, -3.0)

        # var vae
        nn.init.kaiming_normal_(self.model.mu_layer_var.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.model.mu_layer_var.bias, 0.0)
        nn.init.kaiming_normal_(self.model.logvar_layer_var.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.model.logvar_layer_var.bias, -3.0)


    def _initialize_decoder_weights(self):
        for m in self.model.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
                init.constant_(m.bias, 0)



