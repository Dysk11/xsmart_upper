from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).resolve().parent


def pkg_config(*args: str) -> list[str]:
    output = subprocess.check_output(["pkg-config", *args, "opencv4"], text=True)
    return shlex.split(output.strip())


opencv_cflags = pkg_config("--cflags")
opencv_libs = pkg_config("--libs")
include_dirs = [str(ROOT / "native" / "include")]
library_dirs: list[str] = []
libraries = ["rknnrt", "pthread"]
extra_compile_args = ["-O3", "-fvisibility=hidden"]
extra_link_args: list[str] = []

for token in opencv_cflags:
    if token.startswith("-I"):
        include_dirs.append(token[2:])
    else:
        extra_compile_args.append(token)
for token in opencv_libs:
    if token.startswith("-L"):
        library_dirs.append(token[2:])
    elif token.startswith("-l"):
        libraries.append(token[2:])
    else:
        extra_link_args.append(token)

setup(
    name="xsmart-rknn-native",
    version="0.1.0",
    description="RK3588 RKNNRT native perception layer for xsmart_upper",
    ext_modules=[
        Pybind11Extension(
            "xsmart_rknn_native",
            [
                str(ROOT / "native" / "src" / "bindings.cpp"),
                str(ROOT / "native" / "src" / "native_engine.cpp"),
                str(ROOT / "native" / "src" / "rknn_model.cpp"),
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            cxx_std=17,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
