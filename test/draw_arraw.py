#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    # 长度
    scale = np.linalg.norm(pVec_Arr)
    # 单位向量
    pVec_Arr = pVec_Arr / scale
    # z_mat 旋转矩阵，must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))

    qTrans_Mat *= scale
    return qTrans_Mat


def get_arrows(begin_vec, vec, vis):
    z_unit_Arr = np.array([0, 0, 1])
    # 起点、终点
    for i in range(begin_vec.shape[0]):
        begin = begin_vec[i]
        end = np.add(begin, vec[i])
        # 箭头向量、箭头长度
        vec_Arr = np.array(end) - np.array(begin)
        vec_len = np.linalg.norm(vec_Arr)
        # 创建箭头
        mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=0.2 * 1,
            cone_radius=0.06 * 1,
            cylinder_height=0.8 * 1,
            cylinder_radius=0.04 * 1
        )
        mesh_arrow.paint_uniform_color([0, 1, 0])
        mesh_arrow.compute_vertex_normals()
        # 旋转矩阵
        rot_mat = caculate_align_mat(vec_Arr)
        mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
        mesh_arrow.translate(np.array([0, 0, 0]))
        #mesh_arrow.translate(np.array(np.float32(begin[i])))  # 0.5*(np.array(end) - np.array(begin))
        vis.add_geometry(mesh_arrow)
    t = 1


def get_arrow(begin=[0, 0, 0], vec=[0, 0, 1]):
    z_unit_Arr = np.array([0, 0, 1])
    # 起点、终点
    begin = begin
    end = np.add(begin, vec)
    # 箭头向量、箭头长度
    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)
    # 创建坐标系
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])
    # 创建箭头
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * 1,
        cone_radius=0.06 * 1,
        cylinder_height=0.8 * 1,
        cylinder_radius=0.04 * 1
    )
    mesh_arrow.paint_uniform_color([0, 1, 0])
    mesh_arrow.compute_vertex_normals()
    # 创建起点球面
    # mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(radius=10, resolution=20)
    # mesh_sphere_begin.translate(begin)
    # mesh_sphere_begin.paint_uniform_color([0, 1, 1])
    # mesh_sphere_begin.compute_vertex_normals()
    # 创建终点球面
    # mesh_sphere_end = o3d.geometry.TriangleMesh.create_sphere(radius=10, resolution=20)
    # mesh_sphere_end.translate(end)
    # mesh_sphere_end.paint_uniform_color([0, 1, 1])
    # mesh_sphere_end.compute_vertex_normals()
    # 旋转矩阵
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(begin))  # 0.5*(np.array(end) - np.array(begin))
    return mesh_arrow
    # return mesh_frame, mesh_arrow, mesh_sphere_begin, mesh_sphere_end


if __name__ == "__main__":
    mesh_frame, mesh_arrow = get_arrow([0, 0, 0], [0, 1, -300])
    o3d.visualization.draw_geometries(
        geometry_list=[mesh_frame, mesh_arrow],
        window_name="after translate", width=800, height=600
    )
