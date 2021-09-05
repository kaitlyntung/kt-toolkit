
from setuptools import setup, find_packages

setup(
    name='snel_toolkit',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas==1.0.3',
        'scipy==1.4.1',
        'numpy==1.18.1',
        'matplotlib==3.1.3',
        'scikit-learn==0.22.1',
        'h5py==2.10.0'
    ],
    author="Systems Neural Engineering Lab",
)
