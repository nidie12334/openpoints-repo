from setuptools import setup, Extension
import numpy

m_name = "grid_subsampling"

SOURCES = [
    "cpp_utils/cloud/cloud.cpp",
    "grid_subsampling/grid_subsampling.cpp",
    "wrapper.cpp"
]

module = Extension(
    m_name,
    sources=SOURCES,
    include_dirs=[
        numpy.get_include(),  # NumPy 头文件路径
    ],
    extra_compile_args=[
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-fpermissive",  # 降级部分 C++ 严格错误为警告
        "-Wno-error=incompatible-pointer-types",  # 允许 PyObject* 到 PyArrayObject* 的隐式转换
        # 在函数宏层面插入强制转换
        "-DPyArray_NDIM(x)=PyArray_NDIM((PyArrayObject*)x)",
        "-DPyArray_DIM(x,y)=PyArray_DIM((PyArrayObject*)x,y)",
        "-DPyArray_DATA(x)=PyArray_DATA((PyArrayObject*)x)"
    ]
)

setup(
    name=m_name,
    ext_modules=[module],
)









