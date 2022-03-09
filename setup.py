# coding=utf-8
"""Setup script for installing ott as a pip module."""
import os
import setuptools


# Reads the version from ott
__version__ = None
with open('ott/version.py') as f:
  exec(f.read(), globals())


# Reads the requirements from requirements.txt
folder = os.path.dirname(__file__)
path = os.path.join(folder, 'requirements.txt')
install_requires = []
if os.path.exists(path):
  with open(path) as fp:
    install_requires = [line.strip() for line in fp]


setuptools.setup(version=__version__,
                 install_requires=install_requires)
