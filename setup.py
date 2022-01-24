# coding=utf-8
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
