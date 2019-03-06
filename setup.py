# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from distutils.core import setup
from setuptools import find_packages
import os
this = os.path.dirname(__file__)

with open(os.path.join(this, "requirements.txt"), "r") as f:
    requirements = [_ for _ in [_.strip("\r\n ")
                                for _ in f.readlines()] if _ is not None]

packages = find_packages()
assert packages

# read version from the package file.
version_str = '0.2.1'
with (open(os.path.join(this, 'skonnxrt/__init__.py'), "r")) as f:
    line = [_ for _ in [_.strip("\r\n ")
                        for _ in f.readlines()] if _.startswith("__version__")]
    if len(line) > 0:
        version_str = line[0].split('=')[1].strip('" ')

README = os.path.join(os.getcwd(), "README.rst")
with open(README) as f:
    long_description = f.read()
    s = '------------'
    start_pos = long_description.find(s)
    if start_pos >= 0:
        long_description = long_description[start_pos + len(s) + 1:]

setup(
    name='scikit-onnxruntime',
    version=version_str,
    description="Scikit-learn wrapper of onnxruntime",
    long_description=long_description,
    license='MIT License',
    author='Microsoft Corporation',
    author_email='xadupre@microsoft.com',
    url='https://github.com/xadupre/scikit-onnxruntime',
    packages=packages,
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License'],
)
