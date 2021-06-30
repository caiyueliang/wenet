# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
-------------------------------------------------
   Description :
   Author :       caiyueliang
   Date :         2021/06/29
-------------------------------------------------

"""

import os
import sys
from common.version import version as version
from setuptools import setup, find_packages

if sys.version_info < (3, 5):
    raise ValueError(
        'Unsupported Python version %d.%d.%d found. Auto-sklearn requires Python '
        '3.5 or higher.' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )

HERE = os.path.abspath(os.path.dirname(__file__))

setup_reqs = ['Cython', 'numpy']
print("[setup_requires] {}".format(setup_reqs))

with open(os.path.join(HERE, 'requirements.txt')) as fp:
    install_reqs = [r.rstrip() for r in fp.readlines() if not r.startswith('#') and not r.startswith('git+')]
print("[install_requires] {}".format(install_reqs))

print("[version] {}".format(version))


setup(
    name='grpc_client',
    author='di-ml',
    author_email='caiyueliang@qudian.com',
    description='grpc_client',
    version=version,
    keywords=["grpc_client"],
    packages=find_packages(),
    py_modules=['grpc_client'],
    setup_requires=setup_reqs,
    install_requires=install_reqs,
    include_package_data=True,
    license='BSD',
    platforms=['Linux'],
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.5.*',
)
