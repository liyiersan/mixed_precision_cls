import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import timm
import torch
import argparse
import numpy as np
import os.path as osp
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as tordata

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


def cal_metrics(gts, preds):
    if isinstance(gts, list):
        gts = np.concatenate(gts)
    if isinstance(preds, list):
        preds = np.concatenate(preds)
    accuracy = accuracy_score(gts, preds)
    precision = precision_score(gts, preds, average='macro', zero_division=np.nan) # add zero_division to avoid warning
    recall = recall_score(gts, preds, average='macro', zero_division=np.nan)
    f1 = f1_score(gts, preds, average='macro', zero_division=np.nan)
    average = (accuracy + precision + recall + f1) / 4
    dict_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'average': average}
    return dict_metrics

def plot_loss(train_loss_list, test_loss_list, save_dir, save_name='loss.jpg'):
    plt.clf()
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(test_loss_list, label='test_loss')
    plt.legend()
    plt.savefig(osp.join(save_dir, save_name))

def get_data_loader(batch_size, data_dir):
    transform = transforms.ToTensor()
    trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
    train_loader = tordata.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = tordata.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

@torch.no_grad()
def test(model, test_loader, criterion, train_mode='FP32'):
    model.eval()
    mix_precision = (train_mode == 'AMP')
    test_batch_loss_list = []
    gts = []
    preds = []
    start_time = time.time()
    for data, target in test_loader:
        if train_mode == 'FP16':
            data = data.half()
        data, target = data.cuda(), target.cuda()
        with autocast(enabled=mix_precision):
            output = model(data)
            loss = criterion(output, target)
        test_batch_loss_list.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        gts.append(target.cpu().numpy())
        preds.append(pred.cpu().numpy())
    end_time = time.time()
    test_time = end_time - start_time
    print(f'Testing time: {test_time} seconds')
    throughput = len(test_loader.dataset) / test_time
    print(f'Throughput for testing: {throughput} images/sec')
    test_loss = np.mean(test_batch_loss_list)
    dict_metrics = cal_metrics(gts, preds)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Average: {:.4f}'.format(
        test_loss, dict_metrics['accuracy'], dict_metrics['precision'], dict_metrics['recall'], dict_metrics['f1'], dict_metrics['average']))
    model.train()
    return test_loss, dict_metrics


def train(model, train_loader, test_loader, epochs, optimizer, criterion, save_dir, early_stop, train_mode='FP32'):
    model.train()
    mix_precision = (train_mode == 'AMP')
    train_loss_list = []
    test_loss_list = []
    best_test_loss = 1e10
    no_improve = 0
    for epoch in range(epochs):
        print(f'############ Epoch: {epoch} start ###############')
        train_batch_loss_list = []
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if train_mode == 'FP16':
                data = data.half()
            data, target = data.cuda(), target.cuda()
            with autocast(enabled=mix_precision):
                output = model(data)
                loss = criterion(output, target)
            train_batch_loss_list.append(loss.item())
            if mix_precision:
                scale = GradScaler()
                scale.scale(loss).backward()
                scale.step(optimizer)
                scale.update()
            else:
                loss.backward()
                optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(epoch,
                    batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        end_time = time.time()
        train_time = end_time - start_time
        print(f'Training time for epoch {epoch}: {train_time} seconds')
        throughput = len(train_loader.dataset) / train_time
        print(f'Throughput in training for epoch {epoch}: {throughput} images/sec')
        train_loss = np.mean(train_batch_loss_list)
        print('Train set: Average loss: {:.4f}'.format(train_loss))
        test_loss, test_metric =  test(model, test_loader, criterion, train_mode)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            print('Saving model at epoch {} with test loss of {:.4f}'.format(epoch, best_test_loss))
            torch.save(model.state_dict(), osp.join(save_dir, 'best_model.pth'))
        else:
            no_improve += 1
            if no_improve > early_stop:
                print('Early stopping at epoch {}'.format(epoch))
                break
        print() # add a blank line

    model.load_state_dict(torch.load(osp.join(save_dir, 'best_model.pth')))
    return model, train_loss_list, test_loss_list

def main(cfg):
    model_name = cfg.model
    model = getattr(models, model_name)(weights=None, num_classes=200)
    save_dir = osp.join(cfg.save_dir, model_name, cfg.train_mode)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    if cfg.resume:
        if osp.exists(osp.join(save_dir, 'best_model.pth')):
            print('Resume training')
            model.load_state_dict(torch.load(osp.join(save_dir, 'best_model.pth')))
        else:
            print('No model to resume')
    criterion = torch.nn.CrossEntropyLoss()
    valid_mode = ['FP32', 'FP16', 'AMP']
    assert cfg.train_mode in valid_mode, 'train_mode should be one of {}'.format(valid_mode)
    if cfg.train_mode == 'FP16':
        model = model.half()
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader, test_loader = get_data_loader(cfg.batch_size, cfg.data_dir)
    model, train_loss_list, test_loss_list = train(model, train_loader, test_loader, cfg.epochs, optimizer, criterion, save_dir, cfg.early_stop, cfg.train_mode)
    plot_loss(train_loss_list, test_loss_list, save_dir, save_name=f'{cfg.train_mode}_loss.jpg')
    test_loss, test_metric = test(model, test_loader, criterion, cfg.train_mode)
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    for key, value in test_metric.items():
        print('{}: {:.6f}'.format(key, value))
    return test_loss, test_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mixed Precision Training')
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--train_mode', type=str, default='AMP', help='train mode, it should be one of FP32, FP16, AMP')
    parser.add_argument('--resume', type=bool, default=True, help='resume model path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--data_dir', type=str, default='tiny-imagenet-200', help='data directory')
    parser.add_argument('--save_dir', type=str, default='results', help='save directory')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    args = parser.parse_args()
    main(args)
