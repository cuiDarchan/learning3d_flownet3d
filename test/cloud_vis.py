#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" point cloud (.npz) visualization """
import numpy as np
import open3d as o3d

data = np.load('/home/cui/data/kitti_rm_ground/000000.npz')

# open3d 可视化numpy点云
#gt = o3d.geometry.PointCloud()
pos1 = o3d.geometry.PointCloud()
pos2 = o3d.geometry.PointCloud()
#gt.points = o3d.utility.Vector3dVector(data['gt'])
pc = data['pos1']
pos1.points = o3d.utility.Vector3dVector(data['pos1'])
pos2.points = o3d.utility.Vector3dVector(data['pos2'])
#gt.paint_uniform_color([1, 0, 0]) # 绘制颜色，红
pos1.paint_uniform_color([0, 1, 0]) # 绿
pos2.paint_uniform_color([0, 0, 1]) # 蓝
o3d.visualization.draw_geometries([pos1, pos2])


