# snel-toolkit
Implements core neural data processing functionality for many SNEL projects. Add `snel-toolkit` to your project repo as a submodule and install it in your Python 3.7 project environment via `pip install -e snel-toolkit`. If you'll be editing the code, install the pre-commit hooks for code formatting.
# Installation Example
```
# Clone your project repo
git clone git@github.com:snel-repo/my-project.git
# Add snel-toolkit as a submodule
git submodule add git@github.com:snel-repo/snel-toolkit.git
# Create a conda environment for your repo
conda env create --name my-project python=3.7
conda activate my-project
# Install snel-toolkit into your project environment
cd snel-toolkit
pip install -e .
pre-commit install
```
