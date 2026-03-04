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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def test(args, model,test_loader,model_path):

    checkpoint = torch.load(model_path, map_location='cuda:{}'.format(torch.cuda.current_device()))
    model.load_state_dict(checkpoint['model_state_dict'])

    print('start test mode...')
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

            outputs, _, _ = model(inputs, labels, is_train = False)
            # outputs = model(inputs)
            outputs = F.softmax(outputs, dim=-1)
            frame = outputs.shape[0]
            frame_predictions.extend(outputs[:, 1].cpu().tolist())
            frame_labels.extend(labels.expand(frame).cpu().tolist())
            pre = torch.mean(outputs[:, 1])

    frame_results = cal_metrics(frame_labels, frame_predictions, threshold=0.5)
    print(
        'valid result:  Acc: {:.2%}, AUC: {:.4f}, EER: {:.2%}, Precision: {:.2%}, Recall: {:.2%}'.format(
            frame_results.Accuracy,
            frame_results.AUC,
            frame_results.EER,
            frame_results.Precision,
            frame_results.Recall
        )
    )

    # ----------------------------
    # 绘制混淆矩阵
    # ----------------------------
    y_true = np.array(frame_labels)
    y_pred = (np.array(frame_predictions) > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    classes = ['Real(0)', 'Fake(1)']

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Cele-DF Confusion Matrix')

    # 保存图片
    plt.savefig("Cele-DF confusion_matrix.png", dpi=300)
    plt.close()
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device','-dv', type=int, default=0, help="specify which GPU to use")
    #parser.add_argument('--model_path', '-md', type=str, default='models/train/Ce-DF/models_params_12.tar')
    parser.add_argument('--model_path', '-md', type=str, default='models/train/ViT_TS_MoE_DZJ_20/models_params_20_Deepfake.tar')
    parser.add_argument('--resume','-rs', type=int, default=-1, help="which epoch continue to train")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--record_step', type=int, default=100, help="the iteration number to record train state")

    parser.add_argument('--batch_size','-bs', type=int, default=32)
    parser.add_argument('--learning_rate','-lr', type=float, default=1e-3)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()


    start_time = time.time()
    setup_seed(2024)
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.device)
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    test_path = '/data3/law/data/Celeb_DF/test'
    # ----------------------------
    # 1. 定义数据预处理和增强
    # ----------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转为Tensor
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # ----------------------------
    # 2. 初始化数据集和数据加载器
    # ----------------------------
    train_dataset = CustomDataset(csv_file='NeuralTextures_Train.csv', transform=train_transform)
    val_dataset = CustomDataset(csv_file='Deepfake_Val.csv', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    valid_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4)


    model = vit_base_patch16_224_in21k(pretrained=True,num_classes=2)
    model = model.cuda()
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    print('Start eval process...')
    test(args, model,valid_loader,args.model_path)
    duration = time.time()-start_time
    print('The best AUC is {:.2%}'.format(auc))
    print('The run time is {}h {}m'.format(int(duration//3600),int(duration%3600//60)))
