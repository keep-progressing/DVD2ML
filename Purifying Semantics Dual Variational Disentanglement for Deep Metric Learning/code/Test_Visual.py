import torch, math, time, argparse, os
import random, dataset, utils, losses, net
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate
from tqdm import *

if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description=
        'Load pre-trained best checkpoint and evaluate only, no training'
    )
    # 和训练脚本LOG_DIR默认路径对齐
    parser.add_argument('--LOG_DIR',
        default='../logs',
        help = 'Path to log folder where best pth stored'
    )
    parser.add_argument('--dataset',
        default='cub',
        help = 'Dataset: cub, cars, SOP, Inshop'
    )
    parser.add_argument('--embedding-size', default = 512, type = int,
        dest = 'sz_embedding',
        help = 'Embedding dimension'
    )
    parser.add_argument('--batch-size', default = 120, type = int,
        dest = 'sz_batch',
        help = 'Eval batch size'
    )
    parser.add_argument('--nb-classes', default=100, type=int,
                        dest='nb_classes',
                        help='Number of training classes used in training'
                        )
    parser.add_argument('--gpu-id', default = 0, type = int,
        help = 'GPU id'
    )
    parser.add_argument('--workers', default = 4, type = int,
        dest = 'nb_workers',
        help = 'Dataloader workers'
    )
    parser.add_argument('--model', default = 'resnet50',
        help = 'Backbone model'
    )
    parser.add_argument('--loss', default = 'Proxy_Anchor',
        help = 'Loss type (only for path naming, no training)'
    )
    parser.add_argument('--alpha', default = 32, type = float)
    parser.add_argument('--mrg', default = 0.1, type = float)
    parser.add_argument('--IPC', type = int)
    parser.add_argument('--bn-freeze', default = 1, type = int)
    parser.add_argument('--l2-norm', default = 1, type = int)
    parser.add_argument('--remark', default = '', help='suffix for log folder')

    args = parser.parse_args()
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)

    # Log path 和训练脚本完全统一
    LOG_DIR = args.LOG_DIR + '/logs_{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}{}'.format(
        args.dataset, args.model, args.loss, args.sz_embedding, args.alpha,
        args.mrg, "adamw", 1e-4, args.sz_batch, args.remark
    )
    ckpt_path = f"{LOG_DIR}/{args.dataset}_{args.model}_best.pth"
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    # Dataset root
    os.chdir('../data/')
    data_root = os.getcwd()

    # Build eval dataset / dataloader
    if args.dataset != 'Inshop':
        ev_dataset = dataset.load(
            name=args.dataset,
            root=data_root,
            mode='eval',
            transform=dataset.utils.make_transform(
                is_train=False,
                is_inception=(args.model == 'bn_inception')
            ))
        dl_ev = torch.utils.data.DataLoader(
            ev_dataset,
            batch_size=args.sz_batch,
            shuffle=False,
            num_workers=args.nb_workers,
            pin_memory=True
        )
    else:
        query_dataset = Inshop_Dataset(
            root=data_root, mode='query',
            transform=dataset.utils.make_transform(is_train=False, is_inception=(args.model == 'bn_inception'))
        )
        dl_query = torch.utils.data.DataLoader(query_dataset, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers, pin_memory=True)

        gallery_dataset = Inshop_Dataset(
            root=data_root, mode='gallery',
            transform=dataset.utils.make_transform(is_train=False, is_inception=(args.model == 'bn_inception'))
        )
        dl_gallery = torch.utils.data.DataLoader(gallery_dataset, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers, pin_memory=True)

    nb_classes = args.nb_classes

    # Build model skeleton
    if args.model.find('googlenet')+1:
        model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
    elif args.model.find('bn_inception')+1:
        model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
    elif args.model.find('resnet18')+1:
        model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
    elif args.model.find('resnet50')+1:
        model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
    elif args.model.find('resnet101')+1:
        model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
    model = model.cuda()

    if args.gpu_id == -1:
        model = nn.DataParallel(model)

    # -------------------------- 加载增强版checkpoint核心修改 --------------------------
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=f"cuda:{args.gpu_id}")

    # 1. 读取保存的训练参数，做匹配校验，避免参数不一致报错
    saved_args = checkpoint["args"]
    print(f"\n===== Checkpoint Saved Training Config =====")
    print(f"Best Epoch: {checkpoint['epoch']}")
    print(f"Dataset: {saved_args['dataset']}, Embedding Size: {saved_args['sz_embedding']}")
    print(f"Alpha: {saved_args['alpha']}, Margin: {saved_args['mrg']}\n")

    # 关键参数校验，不匹配直接抛出警告
    mismatch = False
    if saved_args["dataset"] != args.dataset:
        print(f"⚠️ Warning: Saved dataset {saved_args['dataset']} != current args {args.dataset}")
        mismatch = True
    if saved_args["sz_embedding"] != args.sz_embedding:
        print(f"⚠️ Warning: Saved embedding {saved_args['sz_embedding']} != current {args.sz_embedding}")
        mismatch = True
    if saved_args["alpha"] != args.alpha or saved_args["mrg"] != args.mrg:
        print(f"⚠️ Warning: Loss hyperparam mismatch between training and current args")
        mismatch = True
    if mismatch:
        input("Press Enter to continue loading anyway or Ctrl+C exit...")

    # 2. 加载模型主干权重
    model.load_state_dict(checkpoint["model_state_dict"])

    # 3. 初始化ProxyAnchor并加载保存的proxy向量（完整复现）
    criterion = None
    if args.loss == 'Proxy_Anchor':
        criterion = losses.Proxy_Anchor(nb_classes=nb_classes, sz_embed=args.sz_embedding, mrg=args.mrg, alpha=args.alpha).cuda()
        proxy_dict = checkpoint.get("proxy_state_dict", None)
        if proxy_dict is not None:
            criterion.load_state_dict(proxy_dict)
            print("✅ ProxyAnchor proxies loaded from checkpoint")

    model.eval()
    print("✅ Model loaded successfully, enter eval mode.")

    # 读取保存的最优指标
    saved_best_recall = checkpoint["best_recall"]
    print(f"\nCheckpoint Record Best R@1: {saved_best_recall[0]*100:.2f}%")
    # ----------------------------------------------------------------------------------

    # Run evaluation
    print("\n===== Start Evaluation =====")
    with torch.no_grad():
        if args.dataset == 'Inshop':
            Recalls = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)
        elif args.dataset != 'SOP':
            Recalls = utils.evaluate_cos(model, dl_ev)
        else:
            Recalls = utils.evaluate_cos_SOP(model, dl_ev)

    # Print results
    print("\n===== Current Evaluation Results =====")
    if args.dataset == 'Inshop':
        ks = [1,10,20,30,40,50]
        for i, k in enumerate(ks):
            print(f"Recall@{k:2d}: {Recalls[i]*100:.2f}%")
    elif args.dataset != 'SOP':
        for i in range(6):
            k = 2**i
            print(f"Recall@{k:2d}: {Recalls[i]*100:.2f}%")
    else:
        for i in range(4):
            k = 10**i
            print(f"Recall@{k:2d}: {Recalls[i]*100:.2f}%")

    # ==============================================
    # Below you can append Grad-CAM visualization code
    # ==============================================
    # from gradcam_utils import GradCAM, overlay_cam
    # target_layer = model.model.layer4[-1] if args.gpu_id != -1 else model.module.model.layer4[-1]
    # gradcam = GradCAM(model, target_layer)
    # ... load single image, generate cam ...