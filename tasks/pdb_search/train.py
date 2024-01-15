import os
import json
from easydict import EasyDict
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import numpy as np

from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist
from tqdm import tqdm

from models.search import Network
from tasks.pdb_search.dataset import PDB_Pair
from tasks.pdb_search.loss import get_loss_func
import utils.transform as T
import utils.pair_transform as PT
import utils

import argparse
parser = argparse.ArgumentParser(description='Arg parser')
parser.add_argument('--max_epoch', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--save_dir', type=str, default='logs/test_log')

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--loss', type=str, default='DefaultLoss')

parser.add_argument('--cfg_path', type=str, default=None)
parser.add_argument('--ins_norm', action='store_true', default=False)
parser.add_argument('--forward_single', action='store_true', default=False)
parser.add_argument('--aug_scale', type=float, default=0.2)
parser.add_argument('--aug_jit_std', type=float, default=0.01)
parser.add_argument('--aug_jit_clip', type=float, default=0.1)

parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--resume_epoch', type=int, default=None)

args = parser.parse_args()


def log_str(msg):
    print(msg)
    with open(os.path.join(args.save_dir, 'test.log'), 'a+') as log_file:
        print(msg, file=log_file)


def compute_loss(in_dict, out_list):
    LOSS = get_loss_func(args.loss)
    loss_func = LOSS()
    return loss_func(in_dict, out_list)


def train_one_epoch(model, loader, optimizer):
    model.train()

    loss_list = []
    for batch in loader:
        optimizer.zero_grad()

        for in_dict in batch:
            in_dict = utils.dict_to_cuda_pair(in_dict)
            out_list = model(in_dict, forward_single=args.forward_single)
            loss = compute_loss(in_dict, out_list)
            loss_list.append(loss.item())

            loss /= len(batch)
            loss.backward()
        
        optimizer.step()
        
    return np.mean(loss_list)


def train(model):
    ## ----- data loader
    train_dataset = PDB_Pair(
        split='train', 
        pdb_swap=True,
        pdb_trans=Compose([
            T.RandomPermutation(keys=['surface', 'atom'])
        ]),
        pair_trans=Compose([
            PT.CenterCoordsPairs(),
            PT.RandomRotationPairs()
        ]),
        post_trans=Compose([
            T.CenterCoords(),
            T.RandomRotation(),
            T.RandomScale(scale_low=1-args.aug_scale, scale_high=1+args.aug_scale),
            T.RandomJitter(std=args.aug_jit_std, clip=args.aug_jit_clip, keys=['surface', 'atom'])
        ])
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=PDB_Pair.collate_fn, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    eval_dataset = PDB_Pair(
        split='eval', 
        pdb_trans=Compose([
            T.RandomPermutation(keys=['surface', 'atom'])
        ]),
        pair_trans=Compose([
            PT.CenterCoordsPairs()
        ]),
        post_trans=Compose([
            T.CenterCoords()
        ])
    )
    eval_loader = DataLoader(
        eval_dataset, 
        collate_fn=PDB_Pair.collate_fn
    )

    ## ----- optimizer
    optimizer, lr_scheduler = utils.get_optimizer(model, args.optim, args.lr, args.max_epoch)

    ## ----- training
    best_loss = 1e5
    best_epoch = 0
    for epoch in tqdm(range(args.max_epoch + 1)):
        epoch_loss = train_one_epoch(model, train_loader, optimizer)
        print('epoch: {}, loss: {:.4f}'.format(epoch, epoch_loss))

        if lr_scheduler:
            lr_scheduler.step()

        if epoch % 2 == 0: # <-- evaluation
            eval_loss = test_one_epoch(model, eval_loader)
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_epoch = epoch

            utils.save_model(model, epoch, args.save_dir)
            print(' -- eval_loss: {:.4f}, best eval_loss: {:.4f}, best epoch: {}.'.format(
                eval_loss, best_loss, best_epoch))


def test_one_epoch(model, loader):
    model.eval()
    
    loss_list = []
    for batch in loader:
        for in_dict in batch:
            with torch.no_grad():
                in_dict = utils.dict_to_cuda_pair(in_dict)
                out_list = model(in_dict)
                loss = compute_loss(in_dict, out_list)
            loss_list.append(loss.item())
    
    loss = np.mean(loss_list)
    return loss


def test_roc_auc(model, loader):
    model.eval()
    
    loss_list = []
    roc_list = []
    pred_list = []
    label_list = []
    for batch in loader:
        for in_dict in batch:
            with torch.no_grad():
                in_dict = utils.dict_to_cuda_pair(in_dict)
                out_list = model(in_dict)
                loss = compute_loss(in_dict, out_list)
            loss_list.append(loss.item())

            xyz1 = in_dict['pdbs'][0]['surface']['xyz'].data.cpu().numpy()
            xyz2 = in_dict['pdbs'][1]['surface']['xyz'].data.cpu().numpy()
            dists = cdist(xyz1, xyz2) < 1.0

            iface_pos1 = dists.sum(1) > 0
            iface_pos2 = dists.sum(0) > 0

            pos_dists1 = dists[iface_pos1, :]
            pos_dists2 = dists[:, iface_pos2]

            desc1_t = out_list[0]['feats_target'].data.cpu().numpy()
            desc1_b = out_list[0]['feats_binder'].data.cpu().numpy()
            desc2_t = out_list[1]['feats_target'].data.cpu().numpy()
            desc2_b = out_list[1]['feats_binder'].data.cpu().numpy()
            
            ## positive pairs
            desc_dists = (np.matmul(desc1_t, desc2_b.T) + np.matmul(desc1_b, desc2_t.T)) / 2
            pos_dists = desc_dists[dists].reshape(-1)

            ## negative pairs
            neg_dists1 = desc_dists[iface_pos1, :][pos_dists1 == 0].reshape(-1)
            neg_dists2 = desc_dists[:, iface_pos2][pos_dists2 == 0].reshape(-1)
            neg_dists = np.concatenate([neg_dists1, neg_dists2], axis=0)
            neg_dists = np.random.choice(neg_dists, len(pos_dists), replace=False)
            
            pos_labels = np.ones_like(pos_dists)
            neg_labels = np.zeros_like(neg_dists)

            preds = np.concatenate([pos_dists, neg_dists])
            labels = np.concatenate([pos_labels, neg_labels])

            roc_list.append(roc_auc_score(labels, preds))
            pred_list.extend(preds.tolist())
            label_list.extend(labels.tolist())

    log_str('Total: {}'.format(len(roc_list)))
    log_str('Mean ROC-AUC: {:.4f}'.format(np.mean(roc_list)))
    log_str('Mean ROC-AUC (overall): {:.4f}'.format(roc_auc_score(label_list, pred_list)))
    log_str('testing loss: {:.4f}'.format(np.mean(loss_list)))


def test(model):
    log_str(args)
    if args.resume_epoch:
        model.load_state_dict(torch.load(
            os.path.join(args.save_dir, 'checkpoints', 'epoch_{}.pth'.format(args.resume_epoch))
        )['state_dict'])
        log_str('load epoch: {}'.format(args.resume_epoch))

    dataset = PDB_Pair(
        split=args.split, 
        pdb_trans=Compose([
            T.RandomPermutation(keys=['surface', 'atom'])
        ]), 
        pair_trans=Compose([
            PT.CenterCoordsPairs()
        ])
    )
    loader = DataLoader(
        dataset,
        collate_fn=PDB_Pair.collate_fn
    )
    test_roc_auc(model, loader)


if __name__ == '__main__':
    print(args)

    ## ----- load model
    if args.cfg_path is not None:
        from shutil import copyfile
        copyfile(
            src=args.cfg_path,
            dst=os.path.join(args.save_dir, 'config.json')
        )
    with open(os.path.join(args.save_dir, 'config.json'), 'r') as f:
        params = json.load(f)
        params = EasyDict(params)
    
    model = Network(params)
    if args.ins_norm:
        model = utils.convert_instance_norm(model)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(' -- total parameters: {:.3f}K'.format(total_params / 1e3))
    
    ## running
    model = model.cuda()
    if args.test:
        test(model)
    else:
        train(model)
