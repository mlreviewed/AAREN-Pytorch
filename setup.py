from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup(
    name='aaren_cuda',
    ext_modules=[
        CUDAExtension(
            name='aaren_cuda',
            sources=['aaren_cuda.cu'], # 
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-gencode=arch=compute_75,code=sm_75']  # Adjust as needed
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
