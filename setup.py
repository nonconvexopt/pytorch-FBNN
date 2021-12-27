from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup


install_requires = [
    'torch>=1.9.0',
    'gpytorch>=1.6.0',
    ]

dependency_links = [
    'git+https://github.com/nonconvexopt/pytorch-SSGE',
    ]

setup(
    name='pytorch_fbnn',
    version='0.1',
    author='Juhyeong Kim',
    author_email='nonconvexopt@gmail.com',
    license_files = ('LICENSE.txt',),
    python_requires='>=3.6',
    install_requires=install_requires,
    dependency_links=dependency_links,
    py_modules=['estimator', 'model'],
    packages=['pytorch_fbnn'],
)