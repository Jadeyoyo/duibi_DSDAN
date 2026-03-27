import argparse
import math
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from DSAN import DSAN
import data_loader
from tsne_same_style import save_dsan_tsne_same_style


def load_data(root_path, src, tar, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    loader_tar = data_loader.load_training(root_path, tar, batch_size, kwargs)
    loader_tar_test = data_loader.load_testing(root_path, tar, batch_size, kwargs)
    return loader_src, loader_tar, loader_tar_test


def get_optimizer(model, args):
    params = [{'params': model.feature_layers.parameters(), 'lr': args.lr_backbone}]
    if getattr(model, 'bottle', None) is not None:
        params.append({'params': model.bottle.parameters(), 'lr': args.lr_bottleneck})
    params.append({'params': model.cls_fc.parameters(), 'lr': args.lr_cls})
    return torch.optim.Adam(params, weight_decay=args.decay)


def adjust_lrs(optimizer, epoch, nepoch, base_lrs):
    factor = math.pow((1 + 10 * (epoch - 1) / nepoch), 0.75)
    for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
        param_group['lr'] = base_lr / factor


def train_epoch(epoch, model, dataloaders, optimizer, args, device):
    model.train()
    source_loader, target_train_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = max(len(source_loader), len(target_train_loader), 1)

    epoch_cls = 0.0
    epoch_lmmd = 0.0
    epoch_loss = 0.0

    warm = epoch <= args.pretrain_epochs

    for i in range(1, num_iter + 1):
        try:
            data_source, label_source = next(iter_source)
        except StopIteration:
            iter_source = iter(source_loader)
            data_source, label_source = next(iter_source)

        try:
            data_target, _ = next(iter_target)
        except StopIteration:
            iter_target = iter(target_train_loader)
            data_target, _ = next(iter_target)

        data_source = data_source.float().to(device)
        data_target = data_target.float().to(device)
        if not isinstance(label_source, torch.Tensor):
            label_source = torch.tensor(label_source)
        label_source = label_source.long().to(device)

        optimizer.zero_grad()

        if warm:
            adaptation_weight = 0.0
            logits, loss_lmmd = model(data_source, data_target, label_source, adaptation_weight=0.0)
        else:
            progress = (epoch - args.pretrain_epochs) / max(args.nepoch - args.pretrain_epochs, 1)
            progress = min(max(progress, 0.0), 1.0)
            adaptation_weight = args.weight * (2.0 / (1.0 + math.exp(-10 * progress)) - 1.0)
            logits, loss_lmmd = model(data_source, data_target, label_source, adaptation_weight=adaptation_weight)

        loss_cls = F.cross_entropy(logits, label_source, label_smoothing=args.label_smoothing)
        loss = loss_cls + adaptation_weight * loss_lmmd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        epoch_cls += float(loss_cls.item())
        epoch_lmmd += float(loss_lmmd.item())
        epoch_loss += float(loss.item())

        if i % args.log_interval == 0 or i == 1 or i == num_iter:
            print(
                f'Epoch: [{epoch:3d}] Iter: [{i:3d}/{num_iter:3d}] '
                f'warm={warm} adapt_w={adaptation_weight:.4f} '
                f'loss={loss.item():.4f} cls={loss_cls.item():.4f} lmmd={loss_lmmd.item():.4f}'
            )

    n = float(num_iter)
    return epoch_loss / n, epoch_cls / n, epoch_lmmd / n


@torch.no_grad()
def evaluate(model, dataloader, device, nclass):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    pred_hist = np.zeros(nclass, dtype=np.int64)
    target_hist = np.zeros(nclass, dtype=np.int64)

    for data, target in dataloader:
        data = data.float().to(device)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)
        target = target.long().to(device)

        logits = model.predict(data)
        total_loss += F.cross_entropy(logits, target, reduction='sum').item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        pred_np = pred.detach().cpu().numpy()
        tar_np = target.detach().cpu().numpy()
        pred_hist += np.bincount(pred_np, minlength=nclass)
        target_hist += np.bincount(tar_np, minlength=nclass)

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    print(f'Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({acc:.2f}%)')
    print('Pred histogram:', pred_hist.tolist())
    print('GT   histogram:', target_hist.tolist())
    return acc


def get_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--src', type=str, default='condition_2')
    parser.add_argument('--tar', type=str, default='condition_3')
    parser.add_argument('--nclass', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nepoch', type=int, default=120)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--weight', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tsne_max_per_class', type=int, default=150)
    parser.add_argument('--tsne_output', type=str, default='result/tsne.png')
    parser.add_argument('--checkpoint_path', type=str, default='model_state.pth')
    parser.add_argument('--lr_backbone', type=float, default=5e-4)
    parser.add_argument('--lr_bottleneck', type=float, default=5e-4)
    parser.add_argument('--lr_cls', type=float, default=5e-4)
    parser.add_argument('--pretrain_epochs', type=int, default=20)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu.lower() != 'cpu' else 'cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataloaders = load_data(args.root_path, args.src, args.tar, args.batch_size)
    model = DSAN(num_classes=args.nclass, bottle_neck=args.bottleneck).to(device)
    optimizer = get_optimizer(model, args)
    base_lrs = [group['lr'] for group in optimizer.param_groups]

    best_acc = -1.0
    stop = 0

    for epoch in range(1, args.nepoch + 1):
        stop += 1
        adjust_lrs(optimizer, epoch, args.nepoch, base_lrs)
        train_loss, train_cls, train_lmmd = train_epoch(epoch, model, dataloaders, optimizer, args, device)
        print(f'Epoch {epoch:3d} summary: train_loss={train_loss:.4f} cls={train_cls:.4f} lmmd={train_lmmd:.4f}')
        acc = evaluate(model, dataloaders[-1], device, args.nclass)

        if acc > best_acc:
            best_acc = acc
            stop = 0
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f'Saved best checkpoint to {args.checkpoint_path}')

        print(f'{args.src}->{args.tar}: best accuracy so far = {best_acc:.2f}%\n')
        if stop >= args.early_stop:
            print(f'Early stop at epoch {epoch}, final best acc={best_acc:.2f}%')
            break

    if os.path.exists(args.checkpoint_path):
        best_model = DSAN(num_classes=args.nclass, bottle_neck=args.bottleneck).to(device)
        state_dict = torch.load(args.checkpoint_path, map_location=device)
        best_model.load_state_dict(state_dict)
        save_dsan_tsne_same_style(args, best_model, dataloaders[-1], args.tsne_output)
        print(f't-SNE saved to: {args.tsne_output}')
    else:
        print('Checkpoint not found, skipped t-SNE export.')
