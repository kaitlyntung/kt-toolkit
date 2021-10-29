
from setuptools import setup, find_packages

# Avoids duplication of requirements
with open('requirements.txt') as file:
    install_requires = file.read().splitlines()

setup(
    name='snel_toolkit',
    author='Systems Neural Engineering Lab',
    version='0.0.0dev',
    install_requires=install_requires,
    packages=find_packages(),
)
