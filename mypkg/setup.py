import os
import sys
from setuptools import find_packages, setup

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if SRC not in sys.path:
    sys.path.append(SRC)

__version__ = '0.1.0'
exec(open(os.path.join(SRC, 'mypkg/version.py')).read())

setup(
    name='mypkg',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    version=__version__,
    description='ML project with cookiecutter',
    author='xk97',
    license='',
)