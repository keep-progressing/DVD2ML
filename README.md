

---

# **Purifying Semantics: Dual Variational Disentanglement for Deep Metric Learning** 

This is the official PyTorch implementation of **Purifying Semantics: Dual Variational Disentanglement for Deep Metric Learning**. This repository provides the source code for experiments conducted on three datasets (CUB-200-2011, Cars-196, Stanford Online Products) and pretrained models with Resnet50 as the backbone.

---

## Requirements

Ensure the following dependencies are installed:

- Python 3
- PyTorch (> 1.0)
- NumPy
- tqdm
- wandb
- [Pytorch-Metric-Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

---

## Datasets

1. Download the following public benchmarks for deep metric learning:
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - Cars-196 ([Images](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotations](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
   - [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)

2. Extract the downloaded `.tgz` or `.zip` files into the `./data/` directory.

---

## Training Embedding Network

### CUB-200-2011

Train an embedding network of Resnet50 (d=512) using **DVD2ML(+PA)**:

```bash
python train.py --dataset cub \
				--embedding-size 512 \
				--batch-size 120 \
				--epochs 40 \
				--nb-classes 100 \
				--gpu-id 1 \
				--workers 4 \
				--model resnet50 \
                --loss Proxy_Anchor \
                --lr 1e-4 \
                --weight-decay 1e-4 \
                --lr-decay-step 5 \
                --lr-decay-gamma 0.5 \
                --alpha 32 \
                --mrg 0.1 \
                --warm 5 \
                --bn-freeze 1 \
                --l2-norm 1
```

| Method | Backbone | R@1 | R@2 | R@4 |
|:------:|:--------:|:---:|:---:|:---:|
| [DVD2ML(+PA)<sup>512</sup>] | Resnet50 | 75.3 | 83.9 | 89.7 |

---

### Cars-196

Train an embedding network of Resnet50 (d=512) using **DVD2ML(+PA)**:

```bash
python train.py --dataset cars \
				--embedding-size 512 \
				--batch-size 80 \
				--epochs 40 \
				--nb-classes 98 \
				--gpu-id 0 \
				--workers 4 \
				--model resnet50 \
                --loss Proxy_Anchor \
                --lr 1e-4 \
                --weight-decay 1e-4 \
                --lr-decay-step 10 \
                --lr-decay-gamma 0.5 \
                --alpha 32 \
                --mrg 0.1 \
                --warm 5 \
                --bn-freeze 1 \
                --l2-norm 1
```

| Method | Backbone | R@1 | R@2 | R@4 |
|:------:|:--------:|:---:|:---:|:---:|
| [DVD2ML(+PA)<sup>512</sup>] | Resnet50 | 91.5 | 94.9 | 97.1 |

---

### Stanford Online Products

Train an embedding network of Resnet50  (d=512) using **DVD2ML(+PA)**:

```bash
python train.py --dataset sop \
				--embedding-size 512 \
				--batch-size 180 \
				--epochs 80 \
				--nb-classes 11318 \
				--gpu-id 0 \
				--workers 4 \
				--model resnet50 \
                --loss Proxy_Anchor \
                --lr 6e-4 \
                --weight-decay 1e-4 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.5 \
                --alpha 32 \
                --mrg 0.1 \
                --warm 5 \
                --bn-freeze 0 \
                --l2-norm 1
```

| Method | Backbone | R@1 | R@10 | R@100 |
|:------:|:--------:|:---:|:----:|:-----:|
| [DVD2ML(+PA)<sup>512</sup>] | Resnet50 | 82.4 | 92.6 | 97.0 |

---

## Acknowledgements

This code is modified and adapted from the following great repositories:

- [Proxy Anchor Loss for Deep Metric Learning](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020)
- [No Fuss Distance Metric Learning using Proxies](https://github.com/dichotomies/proxy-nca)
- [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

---

