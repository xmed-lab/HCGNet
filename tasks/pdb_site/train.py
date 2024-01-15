import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import numpy as np
from easydict import EasyDict

from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist
from tqdm import tqdm

import utils
import utils.transform as T
from tasks.pdb_site.loss import ProteinSegLoss
from tasks.pdb_site.dataset import PDB_Atom
from models.site import Network

import argparse
parser = argparse.ArgumentParser(description='Arg parser')
parser.add_argument('--max_epoch', type=int, default=120)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--save_dir', type=str, default='logs/test_log')

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optim', type=str, default='sgd')

parser.add_argument('--cfg_path', type=str, default=None)
parser.add_argument('--ins_norm', action='store_true', default=False)

parser.add_argument('--aug_scale', type=float, default=0.2)
parser.add_argument('--aug_jit_std', type=float, default=0.01)
parser.add_argument('--aug_jit_clip', type=float, default=0.1)
parser.add_argument('--aug_crop', type=float, default=0.6)

parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--vote', type=int, default=0)
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--resume_epoch', type=str, default=None)

args = parser.parse_args()


def log_str(msg):
    print(msg)
    with open(os.path.join(args.save_dir, 'test.log'), 'a+') as log_file:
        print(msg, file=log_file)


def load_split(split):
    LIST_DIR = './tasks/pdb_site/lists/'
    pdb_ids = np.loadtxt(
        os.path.join(LIST_DIR, f'{split}.txt'), 
        dtype=str
    ).tolist()
    return pdb_ids


def get_split_ids(split):
    if split == 'test':
        return load_split('test')
    
    elif split in ['train_eval', 'train', 'eval']:
        pdb_ids = load_split('train')
        eval_num = int(len(pdb_ids) * 0.1)
        eval_ids = np.arange(0, len(pdb_ids), len(pdb_ids) // eval_num)[:eval_num]

        train_list = []
        eval_list = []
        for i, pid in enumerate(pdb_ids):
            if i in eval_ids:
                eval_list.append(pid)
            else:
                train_list.append(pid)
        
        if split == 'train_eval':
            return train_list, eval_list
        elif split == 'eval':
            return eval_list
        elif split == 'train':
            pick_ids = np.arange(0, len(train_list), len(train_list) // 200)
            train_list = np.array(train_list)[pick_ids].tolist()
            return train_list


def compute_loss(out_dict, in_dict):
    out_cls = out_dict['out_cls']
    in_lbls = in_dict['surface']['iface']
    loss_func = ProteinSegLoss(gamma=0.5)
    loss = loss_func(out_cls, in_lbls)
    return loss


def train_one_epoch(model, loader, optimizer):
    model.train()

    loss_list = []
    for batch in loader:
        optimizer.zero_grad()

        for item in batch:
            item = utils.dict_to_cuda(item)
            out_dict = model(item)
            loss = compute_loss(out_dict, item)
            loss_list.append(loss.item())

            loss /= len(batch)
            loss.backward()

        optimizer.step()

    return np.mean(loss_list)


def train(model):
    ## ----- data loader
    train_list, eval_list = get_split_ids('train_eval')
    train_dataset = PDB_Atom(train_list, transform=Compose([
        T.RandomPermutation(), # 'surf' and 'atom'
        T.CenterCoords(),
        T.RandomRotation(),
        T.RandomScale(scale_low=1-args.aug_scale, scale_high=1+args.aug_scale),
        T.RandomJitter(std=args.aug_jit_std, clip=args.aug_jit_clip, keys=['surface', 'atom']),
        T.RandomCropPointCloud(ratio=args.aug_crop)
    ]))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=PDB_Atom.collate_fn, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    eval_dataset = PDB_Atom(eval_list, transform=Compose([
        T.RandomPermutation(), # 'surf' and 'atom'
        T.CenterCoords()
    ]))
    eval_loader = DataLoader(
        eval_dataset, 
        collate_fn=PDB_Atom.collate_fn
    )

    ## ----- optimizer
    optimizer, lr_scheduler = utils.get_optimizer(model, args.optim, args.lr, args.max_epoch)

    ## ----- training
    best_roc_auc = 0.0
    best_epoch = 0
    for epoch in tqdm(range(args.max_epoch + 1)):
        epoch_loss = train_one_epoch(model, train_loader, optimizer)
        print('epoch: {}, loss: {:.4f}'.format(epoch, epoch_loss))

        if lr_scheduler:
            lr_scheduler.step()

        if epoch % 2 == 0:
            eval_loss, roc_auc = test_one_epoch(model, eval_loader)
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_epoch = epoch

            utils.save_model(model, epoch, args.save_dir)
            print(' -- roc_auc: {:.4f}, eval_loss: {:.4f}, best_roc_auc: {:.4f}, best_epoch: {}.'.format(
                roc_auc, eval_loss, best_roc_auc, best_epoch))


def test_one_epoch(model, loader, verbose=False):
    model.eval()

    if args.vote > 0:
        vote = args.vote
        transform = Compose([
            T.RandomRotation(), # 'surf' and 'atom'
            T.RandomScale(scale_low=1-args.aug_scale, scale_high=1+args.aug_scale),
            T.RandomJitter(std=args.aug_jit_std, clip=args.aug_jit_clip, keys=['surface', 'atom'])
        ])
    else: # without augmentation voting
        vote = 1
        transform = None
    
    mesh_score_votes = []
    mesh_label_votes = []
    loss_list = []
    for v in range(vote):
        mesh_score_list = []
        mesh_label_list = []
        for batch in loader:
            for item in batch:
                if transform:
                    item = transform(item)
                
                item = utils.dict_to_cuda(item)
                with torch.no_grad():
                    out_dict = model(item)
                
                loss = compute_loss(out_dict, item)
                loss_list.append(loss.item())
                out_cls = out_dict['out_cls']
                surf_score = torch.sigmoid(out_cls).data.cpu().numpy()

                surf_xyz = item['surface']['xyz'].data.cpu().numpy()
                mesh_xyz = item['mesh']['xyz'].data.cpu().numpy()

                pdist = cdist(mesh_xyz, surf_xyz)
                pidx = np.argmin(pdist, axis=1)
                mesh_score = surf_score[pidx]

                mesh_score_list.append(mesh_score)
                mesh_label_list.append(item['mesh']['iface'].data.cpu().numpy())
        
        mesh_scores = np.concatenate(mesh_score_list, axis=0)
        mesh_labels = np.concatenate(mesh_label_list, axis=0)
        
        mesh_label_votes.append(mesh_labels)
        mesh_score_votes.append(mesh_scores)

        roc_auc = roc_auc_score(mesh_labels, mesh_scores)
        if verbose:
            log_str('vote {}:, ROC-AUC: {:.8f}'.format(v, roc_auc))
    
    mesh_scores = np.mean(np.array(mesh_score_votes), axis=0)
    mesh_labels = np.mean(np.array(mesh_label_votes), axis=0)
    roc_auc = roc_auc_score(mesh_labels, mesh_scores)

    loss = np.mean(loss_list)
    if verbose:
        log_str('final (voting {}):, ROC-AUC: {:.4f}, loss: {:.4f}'.format(vote, roc_auc, loss))

    return loss, roc_auc


def test(model):
    if args.resume_epoch:
        model.load_state_dict(torch.load(
            os.path.join(args.save_dir, 'checkpoints', 'epoch_{}.pth'.format(args.resume_epoch))
        )['state_dict'])
        log_str('load epoch: {}'.format(args.resume_epoch))
    
    pdb_ids = get_split_ids(args.split)
    pdb_ids = list(set(pdb_ids)) # remove duplicated ids
    dataset = PDB_Atom(pdb_ids, transform=Compose([
        T.RandomPermutation(), # 'surf' and 'atom'
        T.CenterCoords()
    ]))
    loader = DataLoader(
        dataset,
        collate_fn=PDB_Atom.collate_fn
    )
    test_one_epoch(model, loader, verbose=True)


if __name__ == '__main__':
    print(args)

    ## ----- model
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
