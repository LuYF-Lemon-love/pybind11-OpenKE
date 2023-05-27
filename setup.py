# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension("base",
        ["pybind11_ke/base/Base.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        ),
]

setup(
    name="base",
    version=__version__,
    author="LuYF-Lemon-love",
    author_email="3555028709@qq.com",
    url="https://github.com/LuYF-Lemon-love/pybind11-OpenKE",
    description="pybind11-OpenKE 的底层数据处理模块",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)