# pymorphogen
Morphogen Profile Extraction In Python

# Installation
Clone into the repository \
```git clone https://github.com/bennm37/pymorphogen.git```\
Install the package and dependencies \
```pip install -e .```\
Add Higra as a Submodule and build
```git submodule add https://github.com/higra/Higra.git```\
```cd Higra```\
```python setup.py bdist_wheel```\
```cd dist```\
```pip install higra*.whl```
(takes ~20 mins).
Note that numpy scipy and sklearn must be installed for tests to pass.
More info here.
https://higra.readthedocs.io/en/stable/install.html#manual-build