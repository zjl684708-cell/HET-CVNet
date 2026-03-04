## -*- coding: utf-8 -*-
import os, sys
sys.setrecursionlimit(15000)
import torch
import numpy as np
import random
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import time
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import logging
from tqdm import tqdm
import timm
from utils import *
from ViT_TS_MoE import *
from zp import CustomDataset



def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(args, model, optimizer,train_loader,valid_loader,scheduler,save_dir):
    max_accuracy = 0
    global_step = 0
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # checkpoint
    if args.resume > -1:
        checkpoint = torch.load(os.path.join(save_dir, 'models_params_{}.tar'.format(args.resume)),
                                map_location='cuda:{}'.format(torch.cuda.current_device()))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict((checkpoint['optimizer_state_dict']))


    for epoch in range(args.resume+1, args.epochs):
        # train part
        print('start train mode...')
        epoch_loss = 0.0
        total_num = 0
        correct_num = 0
        correct_num_srm = 0
        model.train()

        with torch.enable_grad():
            st_time = time.time()
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs, moe_loss, moe_loss_srm, fal_loss, fal_loss_srm = model(inputs, labels, is_train = True)

                ce_loss = criterion(outputs, labels)
                loss = ce_loss + 1 * moe_loss + 1 * moe_loss_srm + 1 * fal_loss + 1 * fal_loss_srm

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_num += inputs.size(0)

                correct_num += torch.sum(torch.argmax(outputs,1) == labels).item()

                global_step += 1

                # record train stat into tensorboardX
                if global_step % args.record_step ==0:

                    period = time.time() - st_time
                    # train_acc = torch.mean((torch.argmax(outputs,1) == labels).float()).item()
                    log.info('Training state: Epoch [{:0>3}/{:0>3}], Iteration [{:0>3}/{:0>3}], Loss: {:.4f} Acc:{:.2%} time:{}m {}s'
                             .format(epoch+1, args.epochs, i+1, len(train_loader), epoch_loss/(i+1), correct_num/total_num, int(period//60), int(period%60)))
                    st_time = time.time()
                    total_num = 0
                    correct_num = 0
                    correct_num_srm = 0
        # eval part
        print('start eval mode...')
        model.eval()
        video_predictions=[]
        video_labels=[]
        frame_predictions=[]
        frame_labels=[]


        with torch.no_grad():

            for inputs, labels in tqdm(valid_loader, total=len(valid_loader), ncols=70, leave=False, unit='step'):
                inputs = inputs.cuda()
                inputs = inputs.squeeze(0)
                labels = labels.cuda()

                outputs, _, _ = model(inputs, labels, is_train=False)
                # outputs = model(inputs)
                outputs = F.softmax(outputs, dim=-1)
                frame = outputs.shape[0]
                frame_predictions.extend(outputs[:, 1].cpu().tolist())
                frame_labels.extend(labels.expand(frame).cpu().tolist())
                pre = torch.mean(outputs[:, 1])

        frame_results = cal_metrics(frame_labels, frame_predictions, threshold=0.5)
        log.info('valid result: Epoch [{:0>3}/{:0>3}], _Acc: {:.2%}, F_Auc: {:.4} F_EER:{:.2%}'
                 .format(epoch + 1, args.epochs, frame_results.ACC, frame_results.AUC, frame_results.EER))

        # save model
        state = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch}
        torch.save(state, os.path.join(save_dir, 'models_params_{}.tar'.format(epoch)))

        scheduler.step()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device','-dv', type=int, default=0, help="specify which GPU to use")
    parser.add_argument('--model_dir', '-md', type=str, default='models/train')
    #parser.add_argument('--resume','-rs', type=int, default=-1, help="which epoch continue to train")
    parser.add_argument('--resume', '-rs', type=int, default=19, help="which epoch continue to train")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--record_step', type=int, default=100, help="the iteration number to record train state")

    parser.add_argument('--batch_size','-bs', type=int, default=32)
    parser.add_argument('--learning_rate','-lr', type=float, default=3e-5)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()


    start_time = time.time()
    setup_seed(2024)
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.device)
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    save_dir = args.model_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)

    # logging
    if args.resume == -1:
        mode = 'w'
    else:
        mode = 'a'
    logging.basicConfig(
        filename=os.path.join(save_dir, 'train.log'),
        filemode=mode,
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    log.addHandler(handler)
    logging.info(args.model_dir)
    log.info('model dir:' + args.model_dir)

    # ----------------------------
    # 1. 定义数据预处理和增强
    # ----------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # ----------------------------
    # 2. 初始化数据集和数据加载器
    # ----------------------------
    train_dataset = CustomDataset(csv_file='Cele-DF_train.csv', transform=train_transform)
    val_dataset = CustomDataset(csv_file='Cele-DF_val.csv', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    valid_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4)



    model = vit_base_patch16_224_in21k(pretrained=True,num_classes=2)
    model = model.cuda()
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # special defined optim
    special_param = []
    other_param = []
    for name, param in model.named_parameters():
        if 'w_gate' in name or 'w_noise' in name:
            special_param.append(param)
        else:
            other_param.append(param)

    optimizer = optim.Adam([{'params': special_param, 'initial_lr':1e-4}, {'params': other_param, 'initial_lr': args.learning_rate}],
                           lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=args.resume)


    print('Start train process...')
    train(args, model,optimizer,train_loader,valid_loader,scheduler,save_dir)
    duration = time.time()-start_time
    print('The task of {} is completed'.format(args.description))
    print('The best AUC is {:.2%}'.format(auc))
    print('The run time is {}h {}m'.format(int(duration//3600),int(duration%3600//60)))
