import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))
SYSTEM_EIGEN3_DIR = os.environ.get("EIGEN3_INCLUDE_DIR", "/usr/include/eigen3")

if not osp.isdir(SYSTEM_EIGEN3_DIR):
    raise FileNotFoundError(
        f"Eigen3 include directory was not found at '{SYSTEM_EIGEN3_DIR}'. "
        "Install system Eigen3 (e.g. libeigen3-dev) or set EIGEN3_INCLUDE_DIR to a valid include path."
    )



setup(
    name='dpvslam',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('dpvslam_cuda_corr',
            sources=['dpvslam/altcorr/correlation.cpp', 'dpvslam/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        CUDAExtension('dpvslam_cuda_ba',
            sources=['dpvslam/fastba/ba.cpp', 'dpvslam/fastba/ba_cuda.cu', 'dpvslam/fastba/block_e.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            },
            include_dirs=[
                SYSTEM_EIGEN3_DIR]
            ),
        CUDAExtension('dpvslam_lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'dpvslam/lietorch/include'), 
                SYSTEM_EIGEN3_DIR],
            sources=[
                'dpvslam/lietorch/src/lietorch.cpp', 
                'dpvslam/lietorch/src/lietorch_gpu.cu',
                'dpvslam/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3'],}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

