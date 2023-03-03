import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import h5py
import subprocess
import shlex
import json
import glob
from ops import transform_functions, se3
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
import pcl
import open3d as o3d
import math

# 返回所有文件名


def getFlist(path):
    for root, dirs, files in os.walk(path):
        print('root_dir:', root)  # 当前路径
        print('sub_dirs:', dirs)  # 子文件夹
        print('files:', files)  # 文件名称，返回list类型
    return files


def download_modelnet40():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)
	if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
		www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
		zipfile = os.path.basename(www)
		os.system('wget %s; unzip %s' % (www, zipfile))
		os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
		os.system('rm %s' % (zipfile))


def load_data(train, use_normals):
	if train: partition = 'train'
	else: partition = 'test'
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	all_data = []
	all_label = []
	for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
		f = h5py.File(h5_name)
		if use_normals: data = np.concatenate(
		    [f['data'][:], f['normal'][:]], axis=-1).astype('float32')
		else: data = f['data'][:].astype('float32')
		label = f['label'][:].astype('int64')
		f.close()
		all_data.append(data)
		all_label.append(label)
	all_data = np.concatenate(all_data, axis=0)
	all_label = np.concatenate(all_label, axis=0)
	return all_data, all_label


def deg_to_rad(deg):
	return np.pi / 180 * deg


def create_random_transform(dtype, max_rotation_deg, max_translation):
	max_rotation = deg_to_rad(max_rotation_deg)
	rot = np.random.uniform(-max_rotation, max_rotation, [1, 3])
	trans = np.random.uniform(-max_translation, max_translation, [1, 3])
	quat = transform_functions.euler_to_quaternion(rot, "xyz")

	vec = np.concatenate([quat, trans], axis=1)
	vec = torch.tensor(vec, dtype=dtype)
	return vec


def jitter_pointcloud(pointcloud, sigma=0.04, clip=0.05):
	# N, C = pointcloud.shape
	sigma = 0.04*np.random.random_sample()
	pointcloud += torch.empty(pointcloud.shape).normal_(mean=0,
	                          std=sigma).clamp(-clip, clip)
	return pointcloud


def farthest_subsample_points(pointcloud1, num_subsampled_points=768):
	pointcloud1 = pointcloud1
	num_points = pointcloud1.shape[0]
	nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
							 metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
	random_p1 = np.random.random(
	    size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
	idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape(
	    (num_subsampled_points,))
	gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)
	return pointcloud1[idx1, :], gt_mask


def knn_idx(pts, k):
    kdt = cKDTree(pts)
    _, idx = kdt.query(pts, k=k+1)
    return idx[:, 1:]


def get_rri(pts, k):
    # pts: N x 3, original points
    # q: N x K x 3, nearest neighbors
    q = pts[knn_idx(pts, k)]
    p = np.repeat(pts[:, None], k, axis=1)
    # rp, rq: N x K x 1, norms
    rp = np.linalg.norm(p, axis=-1, keepdims=True)
    rq = np.linalg.norm(q, axis=-1, keepdims=True)
    pn = p / rp
    qn = q / rq
    dot = np.sum(pn * qn, -1, keepdims=True)
    # theta: N x K x 1, angles
    theta = np.arccos(np.clip(dot, -1, 1))
    T_q = q - dot * p
    sin_psi = np.sum(np.cross(T_q[:, None], T_q[:, :, None]) * pn[:, None], -1)
    cos_psi = np.sum(T_q[:, None] * T_q[:, :, None], -1)
    psi = np.arctan2(sin_psi, cos_psi) % (2*np.pi)
    idx = np.argpartition(psi, 1)[:, :, 1:2]
    # phi: N x K x 1, projection angles
    phi = np.take_along_axis(psi, idx, axis=-1)
    feat = np.concatenate([rp, rq, theta, phi], axis=-1)
    return feat.reshape(-1, k * 4)


def get_rri_cuda(pts, k, npts_per_block=1):
    import pycuda.autoinit
    mod_rri = SourceModule(open('rri.cu').read() % (k, npts_per_block))
    rri_cuda = mod_rri.get_function('get_rri_feature')

    N = len(pts)
    pts_gpu = gpuarray.to_gpu(pts.astype(np.float32).ravel())
    k_idx = knn_idx(pts, k)
    k_idx_gpu = gpuarray.to_gpu(k_idx.astype(np.int32).ravel())
    feat_gpu = gpuarray.GPUArray((N * k * 4,), np.float32)

    rri_cuda(pts_gpu, np.int32(N), k_idx_gpu, feat_gpu,
             grid=(((N-1) // npts_per_block)+1, 1),
             block=(npts_per_block, k, 1))

    feat = feat_gpu.get().reshape(N, k * 4).astype(np.float32)
    return feat


class UnknownDataTypeError(Exception):
	def __init__(self, *args):
		if args: self.message = args[0]
		else: self.message = 'Datatype not understood for dataset.'

	def __str__(self):
		return self.message


class ModelNet40Data(Dataset):
	def __init__(
		self,
		train=True,
		num_points=1024,
		download=True,
		randomize_data=False,
		use_normals=False
	):
		super(ModelNet40Data, self).__init__()
		if download: download_modelnet40()
		self.data, self.labels = load_data(train, use_normals)
		if not train: self.shapes = self.read_classes_ModelNet40()
		self.num_points = num_points
		self.randomize_data = randomize_data

	def __getitem__(self, idx):
		if self.randomize_data: current_points = self.randomize(idx)
		else: current_points = self.data[idx].copy()

		current_points = torch.from_numpy(
		    current_points[:self.num_points, :]).float()
		label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

		return current_points, label

	def __len__(self):
		return self.data.shape[0]

	def randomize(self, idx):
		pt_idxs = np.arange(0, self.num_points)
		np.random.shuffle(pt_idxs)
		return self.data[idx, pt_idxs].copy()

	def get_shape(self, label):
		return self.shapes[label]

	def read_classes_ModelNet40(self):
		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
		DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
		file = open(os.path.join(
		    DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r')
		shape_names = file.read()
		shape_names = np.array(shape_names.split('\n')[:-1])
		return shape_names


# class ClassificationData(Dataset):
	# def __init__(self, data_class=ModelNet40Data()):
	# 	super(ClassificationData, self).__init__()
	# 	self.set_class(data_class)

	# def __len__(self):
	# 	return len(self.data_class)

	# def set_class(self, data_class):
	# 	self.data_class = data_class

	# def get_shape(self, label):
	# 	try:
	# 		return self.data_class.get_shape(label)
	# 	except:
	# 		return -1

	# def __getitem__(self, index):
	# 	return self.data_class[index]


# class RegistrationData(Dataset):
# 	def __init__(self, algorithm, data_class=ModelNet40Data(), partial_source=False, partial_template=False, noise=False, additional_params={}):
# 		super(RegistrationData, self).__init__()
# 		available_algorithms = ['PCRNet', 'PointNetLK', 'DCP', 'PRNet', 'iPCRNet', 'RPMNet', 'DeepGMR']
# 		if algorithm in available_algorithms: self.algorithm = algorithm
# 		else: raise Exception("Algorithm not available for registration.")

# 		self.set_class(data_class)
# 		self.partial_template = partial_template
# 		self.partial_source = partial_source
# 		self.noise = noise
# 		self.additional_params = additional_params

# 		if self.algorithm == 'PCRNet' or self.algorithm == 'iPCRNet':
# 			from .. ops.transform_functions import PCRNetTransform
# 			self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=1)
# 		if self.algorithm == 'PointNetLK':
# 			from .. ops.transform_functions import PNLKTransform
# 			self.transforms = PNLKTransform(0.8, True)
# 		if self.algorithm == 'RPMNet':
# 			from .. ops.transform_functions import RPMNetTransform
# 			self.transforms = RPMNetTransform(0.8, True)
# 		if self.algorithm == 'DCP' or self.algorithm == 'PRNet':
# 			from .. ops.transform_functions import DCPTransform
# 			self.transforms = DCPTransform(angle_range=45, translation_range=1)
# 		if self.algorithm == 'DeepGMR':
# 			self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri
# 			from .. ops.transform_functions import DeepGMRTransform
# 			self.transforms = DeepGMRTransform(angle_range=90, translation_range=1)
# 			if 'nearest_neighbors' in self.additional_params.keys() and self.additional_params['nearest_neighbors'] > 0:
# 				self.use_rri = True
# 				self.nearest_neighbors = self.additional_params['nearest_neighbors']
# 			else:
# 				self.use_rri = False

# 	def __len__(self):
# 		return len(self.data_class)

# 	def set_class(self, data_class):
# 		self.data_class = data_class

# 	def __getitem__(self, index):
# 		template, label = self.data_class[index]
# 		self.transforms.index = index				# for fixed transformations in PCRNet.
# 		source = self.transforms(template)

# 		# Check for Partial Data.
# 		if self.partial_source: source, self.source_mask = farthest_subsample_points(source)
# 		if self.partial_template: template, self.template_mask = farthest_subsample_points(template)

# 		# Check for Noise in Source Data.
# 		if self.noise: source = jitter_pointcloud(source)

# 		if self.use_rri:
# 			template, source = template.numpy(), source.numpy()
# 			template = np.concatenate([template, self.get_rri(template - template.mean(axis=0), self.nearest_neighbors)], axis=1)
# 			source = np.concatenate([source, self.get_rri(source - source.mean(axis=0), self.nearest_neighbors)], axis=1)
# 			template, source = torch.tensor(template).float(), torch.tensor(source).float()

# 		igt = self.transforms.igt

# 		if self.additional_params['use_masknet']:
# 			if self.partial_source and self.partial_template:
# 				return template, source, igt, self.template_mask, self.source_mask
# 			elif self.partial_source:
# 				return template, source, igt, self.source_mask
# 			elif self.partial_template:
# 				return template, source, igt, self.template_mask
# 		else:
# 			return template, source, igt


class SegmentationData(Dataset):
	def __init__(self):
		super(SegmentationData, self).__init__()

	def __len__(self):
		pass

	def __getitem__(self, index):
		pass


class FlowData(Dataset):
	def __init__(self):
		super(FlowData, self).__init__()
		self.pc1, self.pc2, self.flow = self.read_data()

	def __len__(self):
		if isinstance(self.pc1, np.ndarray):
			return self.pc1.shape[0]
		elif isinstance(self.pc1, list):
			return len(self.pc1)
		else:
			raise UnknownDataTypeError

	def read_data(self):
		pass

	def __getitem__(self, index):
		return self.pc1[index], self.pc2[index], self.flow[index]


class SceneflowDataset(Dataset):
	# def __init__(self, npoints=1024, root='', partition='train'):
	# 	if root == '':
	# 		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	# 		#DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	# 		root = os.path.join(BASE_DIR, os.pardir, 'data')
	# 		#root = os.path.join(DATA_DIR, 'data_processed_maxcut_35_20k_2k_8192')
	# 		if not os.path.exists(root):
	# 			print("To download dataset, click here: https://drive.google.com/file/d/1CMaxdt-Tg1Wct8v8eGNwuT7qRSIyJPY-/view")
	# 			exit()
	# 		else:
	# 			print("SceneflowDataset Found Successfully!")

	# 	self.npoints = npoints
	# 	self.partition = partition
	# 	self.root = root
	# 	if self.partition=='train':
	# 		self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
	# 	else:
	# 		self.datapath = glob.glob(os.path.join(self.root, '*.npz'))
	# 	self.cache = {}
	# 	self.cache_size = 30000

	# 	###### deal with one bad datapoint with nan value
	# 	self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
	# 	######
	# 	print(self.partition, ': ',len(self.datapath))

	# def __getitem__(self, index):
	# 	if index in self.cache:
	# 		pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
	# 	else:
	# 		fn = self.datapath[index]
	# 		with open(fn, 'rb') as fp:
	# 			data = np.load(fp)
	# 			pos1 = data['points1'].astype('float32')
	# 			pos2 = data['points2'].astype('float32')
	# 			color1 = data['color1'].astype('float32')
	# 			color2 = data['color2'].astype('float32')
	# 			flow = data['flow'].astype('float32')
	# 			mask1 = data['valid_mask1']

	# 		if len(self.cache) < self.cache_size:
	# 			self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)

	# 	if self.partition == 'train':
	# 		n1 = pos1.shape[0]
	# 		sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
	# 		n2 = pos2.shape[0]
	# 		sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

	# 		pos1 = pos1[sample_idx1, :]
	# 		pos2 = pos2[sample_idx2, :]
	# 		color1 = color1[sample_idx1, :]
	# 		color2 = color2[sample_idx2, :]
	# 		flow = flow[sample_idx1, :]
	# 		mask1 = mask1[sample_idx1]
	# 	else:
	# 		pos1 = pos1[:self.npoints, :]
	# 		pos2 = pos2[:self.npoints, :]
	# 		color1 = color1[:self.npoints, :]
	# 		color2 = color2[:self.npoints, :]
	# 		flow = flow[:self.npoints, :]
	# 		mask1 = mask1[:self.npoints]

	# 	pos1_center = np.mean(pos1, 0)
	# 	pos1 -= pos1_center
	# 	pos2 -= pos1_center

	# 	return pos1, pos2, color1, color2, flow, mask1

	# def __len__(self):
	# 	return len(self.datapath)

    # kitti data
    def __init__(self, root='', npoints=16384, partition='train'):
        self.npoints = npoints
        self.partition = partition
        if root == '':
            #root = os.path.abspath('/home/cui/data/kitti_rm_ground/')
            root = os.path.abspath('/home/cui/workspace/deepLearning/learning3d/data/')
            if not os.path.exists(root):
                print(
                    "To download dataset, click here: https://drive.google.com/file/d/1CMaxdt-Tg1Wct8v8eGNwuT7qRSIyJPY-/view")
                exit()
            else:
                print("SceneflowDataset Found Successfully!")
        self.cache = {}
        self.cache_size = 30000
        self.root = root
        if self.partition == 'train':
            self.datapath = glob.glob(os.path.join(self.root, 'nuscene_ego_motion/train/', '*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, '*.npz'))

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, color1, color2, flow, mask, r_flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1']
                pos2 = data['pos2']
                flow = data['gt']
                #r_flow = data['r_gt']
                color1 = np.zeros([pos1.shape[0], 3]).astype(np.float32)
                color2 = np.zeros([pos2.shape[0], 3]).astype(np.float32)
                mask = np.ones([pos1.shape[0]]).astype(np.float32)
            # 注意位置
            if len(self.cache) < self.cache_size:
                # self.cache[index] = (pos1, pos2, color1, color2, flow, mask, r_flow)
                self.cache[index] = (pos1, pos2, color1, color2, flow, mask)
        #print("else index= ", index, ", file name: ", self.datapath[index])
        
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        #print( "n1=", n1, " ,n2=", n2)
        # 保留flow中非0的 全部点
        flow_idx = np.unique(np.nonzero(flow))
        no_flow_idx = list(set(range(n1)) - set(flow_idx))
        no_flow_idx = np.array(no_flow_idx)
        #r_flow_idx = np.unique(np.nonzero(r_flow)) # 反向，pos2 到 pos1
        #no_rflow_idx = list(set(range(n2)) - set(r_flow_idx))
        #no_rflow_idx = np.array(no_rflow_idx)
        
        # 保留flow的点，修改随机采样
        # if(flow_idx.shape[0] > int(self.npoints/2)):
        #     # 将一半有场景流的筛选出来，再随机筛选剩下的
        #     flow_idx = np.random.choice(
        #         flow_idx, int(self.npoints/2), replace=False)
        #     sample_idx1 = np.random.choice(
        #         no_flow_idx, self.npoints-flow_idx.shape[0], replace=False)
        # else:
        #     sample_idx1 = np.random.choice(
        #         no_flow_idx, self.npoints-flow_idx.shape[0], replace=False)
        # sample_idx1 = np.array(list(set(flow_idx) | set(sample_idx1)))
        # assert sample_idx1.shape[0] == self.npoints
        # if(r_flow_idx.shape[0] > int(self.npoints/2)):
        #     r_flow_idx = np.random.choice(
        #         r_flow_idx, int(self.npoints/2), replace=False)
        #     sample_idx2 = np.random.choice(
        #         no_rflow_idx, self.npoints-r_flow_idx.shape[0], replace=False)
        # else:
        #     sample_idx2 = np.random.choice(
        #         no_rflow_idx, self.npoints-r_flow_idx.shape[0], replace=False)
        # sample_idx2 = np.array(list(set(r_flow_idx) | set(sample_idx2)))
        # assert sample_idx2.shape[0] == self.npoints
        
        # 原版
        if n1 >= self.npoints:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
        else:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=True)
        if n2 >= self.npoints:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
        else:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=True)
        print("sample_idx1=", sample_idx1.shape[0], " ,sample_idx2=", sample_idx2.shape[0])
        
        pos1 = np.copy(pos1)[sample_idx1, :].astype(np.float32)
        pos2 = np.copy(pos2)[sample_idx2, :].astype(np.float32)
        flow = np.copy(flow)[sample_idx1, :].astype(np.float32)

        color1 = np.zeros([self.npoints, 3]).astype(np.float32)  # 零矩阵
        color2 = np.zeros([self.npoints, 3]).astype(np.float32)  # 零矩阵
        mask = np.ones([self.npoints]).astype(np.float32)       # 1矩阵  
        
        # 重点，进行中心平移
        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center
        # print("shape:", pos1.shape[0], pos2.shape[0], color1.shape[0],
        #       color2.shape[0], flow.shape[0], mask.shape[0])
        return pos1, pos2, color1, color2, flow, mask

    def __len__(self):
        return len(self.datapath)


class SceneflowOwnDataset(Dataset):
    def __init__(self, root='', npoints=16384, partition='train'):
        self.npoints = npoints
        self.partition = partition
        self.file_type = '.pcd'  # ".bin"
        if root == '':
            if(self.file_type == '.bin'):
                root = os.path.abspath('/home/cui/data/nuscenes/v1.0-mini/sweeps/LIDAR_TOP/')
            if(self.file_type == '.pcd'):
                root = os.path.abspath(
                    '/home/cui/data/motion_seg_ht/nuscene_seg/')
            if not os.path.exists(root):
                print(
                    "To download dataset, click here: https://drive.google.com/file/d/1CMaxdt-Tg1Wct8v8eGNwuT7qRSIyJPY-/view")
                exit()
            else:
                print("SceneflowDataset Found Successfully!")
        self.cache = {}
        self.cache_size = 30000
        self.root = root
        self.datapath = []
        self.save_path = os.path.abspath('/home/cui/data/motion_seg_ht/nuscene')
        
        if self.partition == 'train':
            self.datapath = glob.glob(os.path.join(self.root, '*.npz'))
        else:
            self.filenames = getFlist(self.root)
            timestamp_list = []
            # pcd处理
            if (self.file_type == '.pcd'):
                for i, file in enumerate(self.filenames):
                    timestamp = file.strip('.pcd')
                    timestamp_list.append(timestamp)
                timestamp_list.sort()
                for i in range(len(timestamp_list)):
                    self.datapath.append(self.root + '/' + timestamp_list[i] + '.pcd')

            # bin处理
            elif(self.file_type == '.bin'):
                for i, file in enumerate(self.filenames):
                    timestamp = file.strip('.pcd.bin')
                    timestamp = timestamp.split('__')
                    if(timestamp[0]== '008-2018-08-01-15-16-36-0400'):
                        timestamp_list.append(timestamp[2])
                timestamp_list.sort()
                for i in range(len(timestamp_list)):
                    self.datapath.append(
                        self.root + '/'+ 'n008-2018-08-01-15-16-36-0400__LIDAR_TOP__' + timestamp_list[i] + '.pcd.bin')
                
    def __getitem__(self, index):
        delt_t = 1
        if index in self.cache:
            pos1, pos2, color1, color2, flow, mask = self.cache[index]
        else:
            if (self.file_type == '.bin'):
                pos1 = np.fromfile(self.datapath[index], dtype=np.float32)
                pos1 = pos1.reshape((-1, 5))[:, 0:3]
                z_height = np.zeros_like(pos1, dtype=np.float32)
                #z_height[:,2] = 1.5
                pos1 = pos1 + z_height
                pos2 = np.fromfile(self.datapath[index + delt_t], dtype=np.float32)
                pos2 = pos2.reshape((-1, 5))[:, 0:3]
                z_height = np.zeros_like(pos2, dtype=np.float32)
                z_height[:, 2] = 1.5
                #pos2 = pos2 + z_height
                
                # 保存点云
                # cloud1 = o3d.geometry.PointCloud()
                # cloud2 = o3d.geometry.PointCloud()
                # cloud1.points = o3d.utility.Vector3dVector(pos1)
                # cloud2.points = o3d.utility.Vector3dVector(pos2)
                # save_path1 = self.save_path + '/' + str(index)+ '.pcd'
                # save_path2 = self.save_path + '/' + str(index+2000)+ '.pcd'
                # o3d.io.write_point_cloud(save_path1 ,cloud1)
                # o3d.io.write_point_cloud(save_path2 ,cloud2)
    
            elif (self.file_type == '.pcd'):
                pos1 = pcl.load_XYZI(self.datapath[2 * index]).to_array()
                pos2 = pcl.load_XYZI(self.datapath[2 * index + delt_t]).to_array()
                print(self.datapath[2 * index])
                print(self.datapath[2 * index + delt_t])
                flow = pcl.load_XYZI(self.datapath[2 * index]).to_array()
                # 判断非地面点
                ground_index = np.where(np.fabs(pos1[:, 3] - 5.000) < 1e-4)  # 返回索引值
                pos1 = np.delete(pos1, ground_index, axis=0)[:, 0:3]
                ground_index2 = np.where(np.fabs(pos2[:, 3] - 5.000) < 1e-4)  # 返回索引值
                pos2 = np.delete(pos2, ground_index2, axis=0)[:, 0:3]
            flow = pos1
        
        color1 = np.zeros([self.npoints, 3]).astype(np.float32) # 零矩阵
        color2 = np.zeros([self.npoints, 3]).astype(np.float32) # 零矩阵
        mask = np.ones([self.npoints]).astype(np.float32)       # 1矩阵        
        # ** 放在采样之后
        if len(self.cache) < self.cache_size:
            self.cache[index] = (pos1, pos2, color1, color2, flow, mask)  
              
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        if n1 >= self.npoints:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
        else:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=True)
        if n2 >= self.npoints:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
        else:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=True)

        pos1 = np.copy(pos1)[sample_idx1, :].astype(np.float32)
        pos2 = np.copy(pos2)[sample_idx2, :].astype(np.float32)
        flow = np.copy(flow)[sample_idx1, :].astype(np.float32)

        color1 = np.zeros([self.npoints, 3]).astype(np.float32) # 零矩阵
        color2 = np.zeros([self.npoints, 3]).astype(np.float32) # 零矩阵
        mask = np.ones([self.npoints]).astype(np.float32)       # 1矩阵
            
        # 重点，进行中心平移
        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center

        return pos1, pos2, color1, color2, flow, mask

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
	class Data():
		def __init__(self):
			super(Data, self).__init__()
			self.data, self.label = self.read_data()

		def read_data(self):
			return [4,5,6], [4,5,6]

		def __len__(self):
			return len(self.data)

		def __getitem__(self, idx):
			return self.data[idx], self.label[idx]

	cd = RegistrationData('abc')
	import ipdb; ipdb.set_trace()
