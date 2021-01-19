# Copyright 2020 Google LLC.
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

"""Setup script for installing group_testing as a pip module."""
import setuptools

VERSION = '0.1.0'

install_requires = [
    'absl-py>=0.7.0',
    'jax>=0.1.67',
    'jaxlib>=0.1.47',
    'numpy>=1.18.4',
]

description = ('OTT: Optimal Transport Tools in Jax.')


setuptools.setup(
    name='ott',
    version=VERSION,
    packages=setuptools.find_packages(),
    description=description,
    long_description=description,
    url='https://github.com/google-research/ott',
    author='Google LLC',
    author_email='opensource@google.com',
    install_requires=install_requires,
    license='Apache 2.0',
    keywords='optimal transport sinkhorn wasserstein jax',
)
