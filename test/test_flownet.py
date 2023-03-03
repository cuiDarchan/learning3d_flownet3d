#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/cui/workspace/deepLearning/learning3d/')
print(sys.path)
import open3d as o3d
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from models import FlowNet3D
from data_utils import SceneflowDataset
#from data_utils import SceneflowOwnDataset
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import math
from draw_arraw import get_arrows
import time

FLAGS_auto_play = 0

## 非阻塞显示
def f_display_flow(cloud1, cloud2, flow, vis):
    pos1 = o3d.geometry.PointCloud()
    pos2 = o3d.geometry.PointCloud()
    gt = o3d.geometry.PointCloud()
    pos1.points = o3d.utility.Vector3dVector(cloud1)
    pos2.points = o3d.utility.Vector3dVector(cloud2)
    gt.points = o3d.utility.Vector3dVector(flow)
    
    vis.update_geometry(pos1)
    vis.update_geometry(pos2)
    vis.update_geometry(gt)
    vis.poll_events()
    vis.update_renderer()
    
## open3d 显示
def display_open3d(template, source, transformed_source):
    template_ = o3d.geometry.PointCloud()
    source_ = o3d.geometry.PointCloud()
    transformed_source_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template)
    source_.points = o3d.utility.Vector3dVector(source)
    transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
    template_.paint_uniform_color([1, 0, 0]) # 红
    source_.paint_uniform_color([0, 1, 0])
    transformed_source_.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([template_, source_, transformed_source_])

## open3d 显示预测的flow
def display_flow(cloud1, cloud2, predict_flow):
    # get_arrows(cloud1, predict_flow, vis)
    # 绘制线条
    lines_pcd = o3d.geometry.LineSet()
    # 终点, cloud1 作为起点
    end_points = cloud1 + predict_flow
    polygon_points = np.concatenate((cloud1, end_points), axis=0) # 合并多维矩阵
    lines = [[i, cloud1.shape[0]+i] for i in range(cloud1.shape[0])]
    color = [[1, 0, 0] for i in range(cloud1.shape[0])]
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(color)  # 线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    # 绘制第一帧原始点云
    origin_cloud = o3d.geometry.PointCloud()
    origin_cloud.points = o3d.utility.Vector3dVector(cloud1)
    origin_cloud.paint_uniform_color([0, 1, 0])  # 绿
    # 绘制第二帧真实点云
    true_cloud = o3d.geometry.PointCloud()
    true_cloud.points = o3d.utility.Vector3dVector(cloud2)
    true_cloud.paint_uniform_color([0, 0, 1])  # 蓝

    o3d.visualization.draw_geometries([origin_cloud, true_cloud, lines_pcd])

## 显示动静障碍物
def display_static_dym(cloud_t1, cloud_t2, pred_flow):
    cloud1 = o3d.geometry.PointCloud()
    cloud1.points = o3d.utility.Vector3dVector(cloud_t1)
    cloud2 = o3d.geometry.PointCloud()
    cloud2.points = o3d.utility.Vector3dVector(cloud_t2)
    point_index = np.zeros(pred_flow.shape[0], dtype='int32')
    vector = np.zeros(pred_flow.shape[0], dtype='float32')
    t = 6.5  # 动静障碍物阈值
    # 计算每一个vector
    for index in range(pred_flow.shape[0]):
        vector[index] = math.sqrt(pred_flow[index, 0]*pred_flow[index, 0] + pred_flow[index, 1]* pred_flow[index, 1]++ pred_flow[index, 2]* pred_flow[index, 2])
        # 如果vector > t, 则认为是动态，标记为1； 否则认为是静态，标记不变
        if(vector[index]> t):
            point_index[index] = 1

    dym_index = np.argwhere(point_index>0)
    dym = dym_index[:,0].tolist() # [:,0] 将其变成一行元素
    inlier_cloud = cloud1.select_by_index(dym)
    outlier_cloud = cloud1.select_by_index(dym, invert=True) # 设置为True表示保存ind之外的点
    outlier_cloud.paint_uniform_color([0, 0, 1])
    inlier_cloud.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])    

def test_one_epoch(args, net, test_loader):
    net.eval()

    total_loss = 0
    num_examples = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 同时列出索引和数据，进度条
    for i, data in enumerate(tqdm(test_loader)):
        # vis_flag = 4 # 查看的某个帧
        # if i == vis_flag:
        if 1:
            data = [d.to(args.device) for d in data]
            pc1, pc2, color1, color2, flow, mask1 = data
            pc1 = pc1.transpose(2,1).contiguous()
            pc2 = pc2.transpose(2,1).contiguous()
            color1 = color1.transpose(2,1).contiguous()
            color2 = color2.transpose(2,1).contiguous()
            flow = flow
            mask1 = mask1.float()
            #print('mask1:', mask1)
            
            batch_size = pc1.size(0)
            num_examples += batch_size
            flow_pred = net(pc1, pc2, color1, color2).permute(0,2,1)
            loss_1 = torch.mean(mask1 * torch.sum((flow_pred - flow) * (flow_pred - flow), -1) / 2.0)

            pc1, pc2 = pc1.permute(0,2,1), pc2.permute(0,2,1) # 1,2 维度进行交换
            pc1_ = pc1 + flow_pred
            flow_pred_np = flow_pred.detach().cpu().numpy()[0]
            gt = flow.detach().cpu().numpy()[0]
            print("Loss: ", loss_1)
            display_open3d(pc1.detach().cpu().numpy()[0], pc2.detach().cpu().numpy()[0], pc1_.detach().cpu().numpy()[0])
            #display_static_dym(pc1.detach().cpu().numpy()[0], pc2.detach().cpu().numpy()[0], flow_pred.detach().cpu().numpy()[0])
            #display_flow(pc1.detach().cpu().numpy()[0], pc2.detach().cpu().numpy()[0], flow_pred_np)
            #display_flow(pc1.detach().cpu().numpy()[0], pc2.detach().cpu().numpy()[0], gt)
            #f_display_flow(pc1.detach().cpu().numpy()[0], pc2.detach().cpu().numpy()[0], gt, vis)
            total_loss += loss_1.item() * batch_size        
            if FLAGS_auto_play != 0:
                time.sleep(FLAGS_auto_play)
            else:
                at = input("input:")
        
    vis.destroy_window()
    return total_loss * 1.0 / num_examples


def test(args, net, test_loader):
    test_loss = test_one_epoch(args, net, test_loader)

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--model', type=str, default='flownet', metavar='N',
                        choices=['flownet'], help='Model to use, [flownet]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='Point Number [default: 2048]')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Whether to test on unseen category')
    parser.add_argument('--dataset', type=str, default='SceneflowDataset',
                        choices=['SceneflowDataset'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--dataset_path', type=str, default='data_processed_maxcut_35_20k_2k_8192', metavar='N',
                        help='dataset to use')
    parser.add_argument('--pretrained', type=str, default='/home/cui/workspace/deepLearning/learning3d/pretrained/exp_flownet/models/model.best.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda')

    if args.dataset == 'SceneflowDataset':
        # ht数据集
        # test_loader = DataLoader(
        #     SceneflowOwnDataset(npoints=args.num_points, partition='test'),
        #     batch_size=args.test_batch_size, shuffle=False, drop_last=False)

        # kitti数据
        test_loader = DataLoader(
              SceneflowDataset(npoints=args.num_points, partition='test'),
              batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("not implemented")

    net = FlowNet3D()
    assert os.path.exists(args.pretrained), "Pretrained Model Doesn't Exists!"
    net.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    net = net.to(args.device)
        
    test(args, net, test_loader) # 预测，测试
    print('FINISH')


if __name__ == '__main__':
    main()
