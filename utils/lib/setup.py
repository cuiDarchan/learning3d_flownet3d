from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

## pytorch 老版本
# setup(
#     name='pointnet2', #包名
#     ext_modules=[    #构建 C 和 C++ 扩展扩展包
#         CUDAExtension('pointnet2_cuda', [
#             'src/pointnet2_api.cpp',
            
#             'src/ball_query.cpp', 
#             'src/ball_query_gpu.cu',
#             'src/group_points.cpp', 
#             'src/group_points_gpu.cu',
#             'src/interpolate.cpp', 
#             'src/interpolate_gpu.cu',
#             'src/sampling.cpp', 
#             'src/sampling_gpu.cu',
#         ],
#         extra_compile_args={'cxx': ['-g'],
#                             'nvcc': ['-O2']})
#     ],
#     cmdclass={'build_ext': BuildExtension}
# )

## pytorch 新版本
setup(
    name='pointnet2', #包名
    ext_modules=[    #构建 C 和 C++ 扩展扩展包
        CUDAExtension('pointnet2_cuda', [
            'src/pointnet2_api.cpp',
            
            'src_cuda11.3/ball_query.cpp', 
            'src_cuda11.3/ball_query_gpu.cu',
            'src_cuda11.3/group_points.cpp', 
            'src_cuda11.3/group_points_gpu.cu',
            'src_cuda11.3/interpolate.cpp', 
            'src_cuda11.3/interpolate_gpu.cu',
            'src_cuda11.3/sampling.cpp', 
            'src_cuda11.3/sampling_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
