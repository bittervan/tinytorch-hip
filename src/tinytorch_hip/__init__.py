from ._version import __version__
from ._core import add_numpy  # 这是 C++/HIP 编译出来的模块名 _core

__all__ = ["add_numpy", "__version__"]
